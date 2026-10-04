"""Qualitative same-scene FOV/resolution demo:
one identical scene -> front/rear RGB + BEV predictions of 3 models vs GT.
90deg-std | 120deg-std | 120deg-highres, all on bit-identical bev_gt.
"""
import os, sys, cv2, glob, numpy as np, torch
sys.path.insert(0, "E:/dualcam_project/src")
from dualcam_bev_lss import DualCamBEVLSS

ROOT = "E:/dualcam_project"
DATA = f"{ROOT}/data"
CK = f"{ROOT}/checkpoints"
OUT = f"{ROOT}/results/samescene_viz"
os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

COLOR = {0:(0,0,0),1:(0,0,255),2:(255,0,255),3:(80,80,80),4:(0,255,0),5:(0,255,255)}  # BGR
BEV=240; RES,X0,Y0,D0,D1=0.25,-30.0,-30.0,2.0,42.0

def colorize(lab):
    out=np.zeros((*lab.shape,3),np.uint8)
    for k,c in COLOR.items(): out[lab==k]=c
    return out

def fov_mask(fov):
    rows=np.arange(BEV)[:,None]; cols=np.arange(BEV)[None,:]
    X=np.broadcast_to(X0+(BEV-1-rows)*RES,(BEV,BEV)); Y=np.broadcast_to(Y0+cols*RES,(BEV,BEV))
    t=np.tan(np.radians(fov/2))
    fx,fy=X-1.5,Y; fr=(fx>0)&(np.abs(fy)<=np.abs(fx)*t)&(np.hypot(fx,fy)>=D0)&(np.hypot(fx,fy)<=D1)
    rx,ry=X+1.5,Y; rr=(rx<0)&(np.abs(ry)<=np.abs(rx)*t)&(np.hypot(rx,ry)>=D0)&(np.hypot(rx,ry)<=D1)
    return fr|rr

def load(calib_dir, ckpt, ih, iw):
    meta=torch.load(ckpt,map_location="cpu",weights_only=False)
    ds=meta.get("lss_downsample",8); pool=meta.get("lss_pool",0.5)
    m=DualCamBEVLSS.from_calib_file(f"{calib_dir}/calibration.json",num_classes=6,bev_h=240,bev_w=240,
                                    img_h=ih,img_w=iw,downsample=ds,pool_res=pool,pretrained=False)
    st=meta["model_state_dict"] if isinstance(meta,dict) and "model_state_dict" in meta else meta
    m.load_state_dict(st,strict=True); return m.to(dev).eval()

def read(p, ih, iw):
    im=cv2.imread(p); im=cv2.resize(im,(iw,ih))[:,:,::-1].copy().astype(np.float32)/255.0
    return torch.from_numpy(im).permute(2,0,1)

def pred(m, dsdir, name, ih, iw):
    imgs=torch.stack([read(f"{dsdir}/front_rgb/{name}",ih,iw),
                      read(f"{dsdir}/rear_rgb/{name}",ih,iw)],0).unsqueeze(0).to(dev)
    with torch.no_grad(): p=torch.argmax(m(imgs)[0],0).cpu().numpy().astype(np.uint8)
    return p

# models: (label, calib/data dir, ckpt, img_h, img_w, fov)
D90=f"{DATA}/carla_dualcam_t01_samescene_90"
D120=f"{DATA}/carla_dualcam_t01_samescene_120"
D120HR=f"{DATA}/carla_dualcam_t01_samescene_120hr"
M = [
 ("90 standard",  D90,   f"{CK}/dualcam_bev_lss_m500_90_best.pt",  224,400, 90),
 ("120 standard", D120,  f"{CK}/dualcam_bev_lss_m500_120_best.pt", 224,400, 120),
 ("120 HIGH-RES", D120HR,f"{CK}/dualcam_bev_lss_m500_120hr_best.pt",448,800, 120),
]
models=[(lbl, load(d,ck,ih,iw), d, ih, iw, fov) for lbl,d,ck,ih,iw,fov in M]

# pick frames with most GT vehicle pixels (most illustrative)
files=sorted(os.path.basename(p) for p in glob.glob(f"{D120}/bev_gt/*.png"))
scored=[]
for n in files[::5]:  # sample every 5th for speed
    g=cv2.imread(f"{D120}/bev_gt/{n}",cv2.IMREAD_GRAYSCALE)
    scored.append((int((g==1).sum()), n))
scored.sort(reverse=True)
picks=[n for _,n in scored[:6]]
print("picked frames:", picks)

def panel_bev(img_color, title, fov=None):
    P=480
    v=cv2.resize(img_color,(P,P),interpolation=cv2.INTER_NEAREST)
    if fov is not None:
        mask=fov_mask(fov).astype(np.uint8)*255
        mask=cv2.resize(mask,(P,P),interpolation=cv2.INTER_NEAREST)
        cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(v,cnts,-1,(255,255,255),1)
    cv2.circle(v,(P//2,P//2),5,(255,255,255),-1)  # ego
    cv2.putText(v,"FRONT",(P//2-34,18),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    cv2.rectangle(v,(0,0),(P-1,P-1),(120,120,120),1)
    bar=np.zeros((30,P,3),np.uint8)
    cv2.putText(bar,title,(8,21),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
    return np.vstack([bar,v])

for name in picks:
    # inputs: 120hr front/rear (nicest), resized to 480 wide keeping aspect
    fr=cv2.resize(cv2.imread(f"{D120HR}/front_rgb/{name}"),(480,270))
    re=cv2.resize(cv2.imread(f"{D120HR}/rear_rgb/{name}"),(480,270))
    cv2.putText(fr,"FRONT 120",(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
    cv2.putText(re,"REAR 120",(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
    inrow=np.hstack([fr,re])  # 960x270

    gt=cv2.imread(f"{D120}/bev_gt/{name}",cv2.IMREAD_GRAYSCALE)
    panels=[panel_bev(colorize(gt),"Ground Truth",None)]
    for lbl,m,d,ih,iw,fov in models:
        p=pred(m,d,name,ih,iw)
        panels.append(panel_bev(colorize(p),lbl,fov))
    bevrow=np.hstack(panels)  # 4*480 wide

    # center the input row over the bev row width
    W=bevrow.shape[1]
    pad=np.zeros((inrow.shape[0], (W-inrow.shape[1])//2, 3),np.uint8)
    inrow_c=np.hstack([pad,inrow,pad])
    if inrow_c.shape[1]<W: inrow_c=np.hstack([inrow_c,np.zeros((inrow_c.shape[0],W-inrow_c.shape[1],3),np.uint8)])
    gap=np.zeros((12,W,3),np.uint8)
    full=np.vstack([inrow_c,gap,bevrow])
    outp=f"{OUT}/{name.replace('.png','')}_panel.png"
    cv2.imwrite(outp,full)
    print("wrote",outp, full.shape)
print("DONE")
