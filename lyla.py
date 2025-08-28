# Hi if you see this, its a small early form of lyla
# 
# python lyla.py "your query"

import os, re, math, sys, requests, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 6000
    d_model = 192
    n_heads = 3
    d_ff = 384
    n_enc = 2
    n_dec = 2
    max_len = 512
    pad_id, bos_id, eos_id, unk_id = 0, 1, 2, 3
    searx_url = os.environ.get("SEARX_URL", "http://localhost:8888/search")
    top_k_results = 5
    extractive_sentences = 4
    gen_max_len = 140

class HashTok:
    def __init__(self, vocab, pad=0, bos=1, eos=2, unk=3):
        self.vocab=vocab; self.pad=pad; self.bos=bos; self.eos=eos; self.unk=unk
        self._rx_ws = re.compile(r"\s+")
    def _tok(self, s:str):
        s=s.strip().lower()
        s=re.sub(r"[^a-z0-9\s.,;:!?'-]", " ", s)
        return [t for t in self._rx_ws.split(s) if t]
    def _id(self,t:str):
        if t=="<pad>": return self.pad
        if t=="<bos>": return self.bos
        if t=="<eos>": return self.eos
        return 4 + (hash(t) % (self.vocab-4))
    def encode(self,s,add_bos=False,add_eos=True,max_len=CFG.max_len):
        ids=[self._id(t) for t in self._tok(s)]
        if add_bos: ids=[self.bos]+ids
        if add_eos: ids=ids+[self.eos]
        return ids[:max_len]
    def encode_lex(self,s,add_bos=True,add_eos=True,max_len=CFG.max_len):
        toks=(["<bos>"] if add_bos else []) + self._tok(s) + (["<eos>"] if add_eos else [])
        ids=[]; lex={self.pad:"<pad>", self.bos:"<bos>", self.eos:"<eos>", self.unk:"<unk>"}
        for t in toks:
            i=self._id(t)
            if i not in lex and t not in ("<bos>","<eos>","<pad>"): lex[i]=t
            ids.append(i)
        return ids[:max_len], lex
    def decode(self, ids, lex):
        out=[]
        for i in ids:
            if i in (CFG.pad_id, CFG.bos_id): continue
            if i==CFG.eos_id: break
            out.append(lex.get(i, "<unk>"))
        s=" ".join(out)
        s=re.sub(r"\s+([.,;:!?'])", r"\1", s)
        s=re.sub(r"\s{2,}", " ", s).strip()
        return s
    def pad_batch(self, seqs):
        m=max(len(x) for x in seqs)
        out=np.full((len(seqs),m), CFG.pad_id, dtype=np.int64)
        for i,x in enumerate(seqs): out[i,:len(x)]=np.array(x, dtype=np.int64)
        return torch.tensor(out)

tok=HashTok(CFG.vocab_size)

def sinusoid_pe(L,d):
    pe=torch.zeros(L,d)
    pos=torch.arange(0,L,dtype=torch.float).unsqueeze(1)
    div=torch.exp(torch.arange(0,d,2).float()*(-math.log(10000.0)/d))
    pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
    return pe

def scaled_dot_attn(Q,K,V,mask=None):
    d=Q.size(-1); logits=Q @ K.transpose(-2,-1)/math.sqrt(d)
    if mask is not None: logits=logits.masked_fill(mask, float('-inf'))
    w=F.softmax(logits, dim=-1)
    return w @ V, w

class MHA(nn.Module):
    def __init__(self,d_model,n_heads):
        super().__init__(); self.h=n_heads; self.d=d_model; self.dk=d_model//n_heads
        self.Wq=nn.Linear(d_model,d_model,bias=False)
        self.Wk=nn.Linear(d_model,d_model,bias=False)
        self.Wv=nn.Linear(d_model,d_model,bias=False)
        self.Wo=nn.Linear(d_model,d_model,bias=False)
    def split(self,x):
        B,T,D=x.size(); return x.view(B,T,self.h,self.dk).transpose(1,2)
    def merge(self,x):
        B,h,T,d=x.size(); return x.transpose(1,2).contiguous().view(B,T,h*d)
    def forward(self,x_q,x_kv,mask=None):
        Q=self.split(self.Wq(x_q)); K=self.split(self.Wk(x_kv)); V=self.split(self.Wv(x_kv))
        out,_=scaled_dot_attn(Q,K,V,mask); return self.Wo(self.merge(out))

class FFN(nn.Module):
    def __init__(self,d_model,d_ff):
        super().__init__(); self.fc1=nn.Linear(d_model,d_ff); self.fc2=nn.Linear(d_ff,d_model)
    def forward(self,x): return self.fc2(F.gelu(self.fc1(x)))

class EncLayer(nn.Module):
    def __init__(self,d_model,n_heads,d_ff):
        super().__init__(); self.sa=MHA(d_model,n_heads); self.ln1=nn.LayerNorm(d_model)
        self.ff=FFN(d_model,d_ff); self.ln2=nn.LayerNorm(d_model)
    def forward(self,x,pad_mask=None):
        x=x+self.sa(self.ln1(x), self.ln1(x), pad_mask)
        x=x+self.ff(self.ln2(x))
        return x

class DecLayer(nn.Module):
    def __init__(self,d_model,n_heads,d_ff):
        super().__init__(); self.msa=MHA(d_model,n_heads); self.ln1=nn.LayerNorm(d_model)
        self.csa=MHA(d_model,n_heads); self.ln2=nn.LayerNorm(d_model)
        self.ff=FFN(d_model,d_ff); self.ln3=nn.LayerNorm(d_model)
    def forward(self,y,mem,look_mask=None,mem_mask=None):
        y=y+self.msa(self.ln1(y), self.ln1(y), look_mask)
        y=y+self.csa(self.ln2(y), mem, mem_mask)
        y=y+self.ff(self.ln3(y))
        return y

class TinyBART(nn.Module):
    def __init__(self,cfg:CFG):
        super().__init__(); self.cfg=cfg
        self.emb=nn.Embedding(cfg.vocab_size,cfg.d_model,padding_idx=cfg.pad_id)
        self.emb_dec=self.emb
        self.pe=sinusoid_pe(cfg.max_len,cfg.d_model).unsqueeze(0)
        self.enc_layers=nn.ModuleList([EncLayer(cfg.d_model,cfg.n_heads,cfg.d_ff) for _ in range(cfg.n_enc)])
        self.dec_layers=nn.ModuleList([DecLayer(cfg.d_model,cfg.n_heads,cfg.d_ff) for _ in range(cfg.n_dec)])
        self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False); self.lm_head.weight=self.emb.weight
        self.ln_enc=nn.LayerNorm(cfg.d_model); self.ln_dec=nn.LayerNorm(cfg.d_model)
    def enc(self,x,pad_mask=None):
        B,T=x.size(); h=self.emb(x)+self.pe[:,:T,:].to(x.device)
        for lyr in self.enc_layers: h=lyr(h,pad_mask)
        return self.ln_enc(h)
    def dec(self,y,mem,look_mask=None,mem_mask=None):
        B,T=y.size(); h=self.emb_dec(y)+self.pe[:,:T,:].to(mem.device)
        for lyr in self.dec_layers: h=lyr(h,mem,look_mask,mem_mask)
        return self.ln_dec(h)
    def forward(self,src,tgt,src_pad_mask=None,tgt_look_mask=None,mem_mask=None):
        mem=self.enc(src,src_pad_mask); out=self.dec(tgt,mem,tgt_look_mask,mem_mask); return self.lm_head(out)
    @torch.no_grad()
    def generate(self,src,max_len=CFG.gen_max_len):
        self.eval(); device=next(self.parameters()).device
        src_mask=(src==CFG.pad_id).unsqueeze(1).unsqueeze(2)
        mem=self.enc(src.to(device), src_mask.to(device))
        ys=torch.full((src.size(0),1), CFG.bos_id, dtype=torch.long, device=device)
        for _ in range(max_len):
            T=ys.size(1)
            look=torch.triu(torch.ones(T,T,device=device),1).bool().unsqueeze(0).unsqueeze(1)
            out=self.dec(ys,mem,look,src_mask.to(device))
            logits=self.lm_head(out[:,-1:,:])
            nxt=torch.argmax(logits,dim=-1)
            ys=torch.cat([ys,nxt],dim=1)
            if (nxt==CFG.eos_id).all(): break
        return ys

def pad_mask_from_ids(ids): return (ids==CFG.pad_id).unsqueeze(1).unsqueeze(2)

def searx_search(q,k=CFG.top_k_results,url=CFG.searx_url):
    try:
        r=requests.get(url, params={"q":q,"format":"json"}, timeout=10)
        r.raise_for_status(); j=r.json()
        return j.get("results",[])[:k]
    except: return []

def split_sentences(t):
    t=re.sub(r"\s+"," ",t)
    s=re.split(r'(?<=[\.\!\?])\s', t)
    return [x.strip() for x in s if x.strip()]

def extractive_summary(snips, q, n=CFG.extractive_sentences):
    S=[]
    for r in snips:
        s=r.get("content") or r.get("snippet") or ""
        S.extend(split_sentences(s))
    if not S: return ""
    vocab={}
    for sent in S:
        for w in re.findall(r"[a-z0-9']+", sent.lower()):
            vocab[w]=vocab.get(w,0)+1
    qset=set(re.findall(r"[a-z0-9']+", q.lower()))
    scores=[]
    for sent in S:
        ws=re.findall(r"[a-z0-9']+", sent.lower())
        if not ws: scores.append(0.0); continue
        tf={}; [tf.setdefault(w,0) or None for w in ws]
        for w in ws: tf[w]+=1
        sc=0.0
        for w,c in tf.items():
            idf=math.log(1+(len(S)/(1+vocab.get(w,1))))
            sc += (c/len(ws))*idf*(1.5 if w in qset else 1.0)
        scores.append(sc)
    idx=np.argsort(scores)[::-1][:n]
    picks=[S[i] for i in sorted(idx)]
    return " ".join(picks)

def bart_paraphrase(model: TinyBART, text:str):
    ids,lex=tok.encode_lex(text,add_bos=True,add_eos=True,max_len=CFG.max_len)
    src=tok.pad_batch([ids]).to(CFG.device)
    out=model.generate(src,max_len=min(CFG.gen_max_len,len(ids)+40))[0].tolist()
    dec=tok.decode(out,lex)
    if len(dec.split())<6: return text
    return dec

def summarize_query(q:str, use_bart=True):
    res=searx_search(q)
    if not res: return "no results"
    para=extractive_summary(res,q,n=CFG.extractive_sentences)
    if not para:
        titles=[r.get("title","") for r in res[:3] if r.get("title")]
        para=". ".join(titles) if titles else "no text"
    if use_bart:
        model=TinyBART(CFG).to(CFG.device); model.eval()
        with torch.no_grad(): para=bart_paraphrase(model,para)
    return para

if __name__=="__main__":
    q=" ".join(sys.argv[1:]) if len(sys.argv)>1 else input("query> ").strip()
    print(summarize_query(q, use_bart=True))
