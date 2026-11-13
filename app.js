'use strict';
(() => {
  const $ = s => document.querySelector(s), canvas = $('#dynamics'), scene=document.createElement('canvas');
  scene.width=1200;scene.height=680;const ctx=scene.getContext('2d'),display=canvas.getContext('2d');
  const C = {ink:'#12243b', blue:'#176cc4', coral:'#ed7159', rule:'#dce6f2', soft:'#f3f7fc'};
  const cache = new Map();
  let mode='manifold', kind='hd', selected=[], progress=0, playing=false, elapsed=0, last=0, serial=0;
  const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
  function text(s,x,y,size=16,color=C.ink,align='left',bold=false) {
    ctx.fillStyle=color;ctx.font=`${bold?600:400} ${Math.max(16,size)}px Arial,sans-serif`;ctx.textAlign=align;ctx.fillText(s,x,y);
  }
  function line(points,color=C.ink,width=2,dash=[]) {
    if(!points.length)return;ctx.beginPath();points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([]);
  }
  function dot(x,y,r,fill,stroke=null,lw=1) {
    ctx.beginPath();ctx.arc(x,y,r,0,2*Math.PI);if(fill){ctx.fillStyle=fill;ctx.fill()}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=lw;ctx.stroke()}
  }
  function color(v) {
    const end=v<0?[36,113,190]:[231,95,64],base=[246,248,252],q=Math.min(1,Math.abs(v));
    return `rgb(${end.map((v,j)=>Math.round(base[j]+q*(v-base[j]))).join(',')})`;
  }
  const frame=d=>Math.round(progress*(d.arrays.time.length-1));
  function row(d,key,i=frame(d)) {
    const shape=d.meta.arrays[key].shape,n=shape.slice(1).reduce((a,b)=>a*b,1);return d.arrays[key].subarray(i*n,(i+1)*n);
  }
  function average(d,key,multiplier=1) {
    const spec=d.meta.arrays[key],n=spec.shape[0],width=spec.shape.slice(1).reduce((a,b)=>a*b,1),out=new Float32Array(n);
    for(let i=0;i<n;i++){let sum=0;for(let j=0;j<width;j++)sum+=d.arrays[key][i*width+j];out[i]=sum/width*multiplier}return out;
  }
  function numeric(v) {
    const a=Math.abs(v);return a>=1000?v.toExponential(1):a>=10?v.toFixed(0):a>=1?v.toFixed(1):v.toFixed(2);
  }
  function chart(d,series,x,y,w,h,{labels=[],zero=false,unit='Time (s)',decimals=null}={}) {
    let lo=Infinity,hi=-Infinity;for(const s of series)for(const v of s){lo=Math.min(lo,v);hi=Math.max(hi,v)}
    if(zero){lo=Math.min(0,lo);hi=Math.max(0,hi)}if(hi-lo<1e-8)hi=lo+1;
    const gap=(hi-lo)*.08;if(!zero||lo<0)lo-=gap;hi+=gap;
    const t=d.arrays.time,imax=frame(d),end=t[t.length-1],fy=v=>y+h-(v-lo)/(hi-lo)*h;
    for(let j=0;j<3;j++){const v=lo+(hi-lo)*j/2,yy=fy(v);line([[x,yy],[x+w,yy]],C.rule,.8);text(decimals===null?numeric(v):v.toFixed(decimals),x-13,yy+4,13,C.ink,'right')}
    if(lo<0&&hi>0)line([[x,fy(0)],[x+w,fy(0)]],'#b6c8da',1);
    series.forEach((s,j)=>{const pts=Array.from(s,(v,i)=>[x+t[i]/end*w,fy(v)]);line(pts,'#dbe4ef',1.1);line(pts.slice(0,imax+1),j?C.coral:C.blue,2.8);dot(...pts[imax],3.7,j?C.coral:C.blue)});
    for(const q of [0,.5,1])text(String(Math.round(q*end*10)/10),x+q*w,y+h+24,13,C.ink,'center');text(unit,x+w/2,y+h+49,14,C.ink,'center');
    labels.forEach((s,j)=>{line([[x+w-190+j*105,y-19],[x+w-170+j*105,y-19]],j?C.coral:C.blue,2.5);text(s,x+w-164+j*105,y-14,12)});
  }
  function referenceBounds(d) {
    const dims=d.meta.arrays.cloud.shape[2],vals=d.arrays.cloud,min=Array(dims).fill(Infinity),max=Array(dims).fill(-Infinity);
    for(let i=0;i<vals.length;i++){const j=i%dims;min[j]=Math.min(min[j],vals[i]);max[j]=Math.max(max[j],vals[i])}
    if(d.arrays.reference_cloud)for(let i=0;i<d.arrays.reference_cloud.length;i++){const j=i%dims;min[j]=Math.min(min[j],d.arrays.reference_cloud[i]);max[j]=Math.max(max[j],d.arrays.reference_cloud[i])}
    return {min,max,center:min.map((v,j)=>(v+max[j])/2),range:Math.max(...max.map((v,j)=>v-min[j]))};
  }
  function cloudView(d,x,width) {
    const hd=d.meta.kind==='hd',n=d.meta.arrays.cloud.shape[1],q=row(d,'cloud'),dim=hd?2:3;
    const b=d.bounds,cx=x+width/2,cy=306,scale=hd?Math.min(width*.76,340)/b.range:Math.min(width*.80,410)/b.projectedRange;
    text(hd?'Head-direction ring':'Grid-cell torus',x+25,68,23,C.ink,'left',true);
    text(`${n.toLocaleString()} population states`,x+25,99,15);
    text(`${d.arrays.time[frame(d)].toFixed(2)} s`,x+width-25,97,15,C.blue,'right',true);
    function xy(p) {
      if(hd)return [cx+(p[0]-b.center[0])*scale,cy-(p[1]-b.center[1])*scale,0];
      const xx=p[0]-b.center[0],yy=p[1]-b.center[1],zz=p[2]-b.center[2],az=-.72,el=.96;
      return [cx+(xx*Math.cos(az)-yy*Math.sin(az))*scale,cy-((xx*Math.sin(az)+yy*Math.cos(az))*Math.sin(el)+zz*Math.cos(el))*scale,(xx*Math.sin(az)+yy*Math.cos(az))*Math.cos(el)-zz*Math.sin(el)];
    }
    // A small coordinate triad identifies the dimensionality without obscuring the point cloud.
    const tx=x+49,ty=488,origin=xy(b.center);
    for(let j=0;j<dim;j++){
      const unit=b.center.slice();unit[j]+=1;const p=xy(unit),dx=p[0]-origin[0],dy=p[1]-origin[1],length=Math.hypot(dx,dy),ex=tx+dx/length*39,ey=ty+dy/length*39;
      line([[tx,ty],[ex,ey]],'#8ea4bc',1.2);text(['z₁','z₂','z₃'][j],ex+dx/length*14,ey+dy/length*12+5,16,C.ink,'center');
    }
    if(d.arrays.reference_cloud){const ref=d.arrays.reference_cloud;for(let j=0;j<ref.length/dim;j++){const p=xy(ref.subarray(j*dim,(j+1)*dim));dot(p[0],p[1],hd?1.2:1.0,'#c7d5e45c')}}
    const points=[];for(let j=0;j<n;j++){const p=xy(q.subarray(j*dim,(j+1)*dim));points.push([...p,j])}points.sort((a,b)=>a[2]-b[2]);
    for(const p of points)dot(p[0],p[1],hd?2.15:2.05,hd?'#176cc47d':'#176cc4a0');
    text('Fixed linear readout of population activity',cx,568,14,C.ink,'center');
    text(hd?'HD recovery: 25 s':'Grid recovery: 200 s',cx,602,15,C.ink,'center',true);
  }
  function hdBump(d) {
    const h=row(d,'activity'),cx=273,cy=255,r=136;
    dot(cx,cy,r,null,'#d3dfec',1.5);
    for(let j=0;j<h.length;j++){const a=-Math.PI+j*2*Math.PI/h.length;dot(cx+r*Math.cos(a),cy-r*Math.sin(a),6.8,color(h[j]),'#ffffff',.7)}
    text('90°',cx,cy-r-25,14,C.ink,'center');text('−90°',cx,cy+r+31,14,C.ink,'center');text('180°',cx-r-24,cy+5,14,C.ink,'right');text('0°',cx+r+26,cy+5,14);
    text('64 neurons',cx,cy-3,20,C.ink,'center',true);text('preferred direction',cx,cy+23,14,C.ink,'center');
    const x=660,y=160,w=475,hh=207,lo=d.activityRange[0],hi=d.activityRange[1],fy=v=>y+hh-(v-lo)/(hi-lo)*hh;
    text('Population activity',x,y-36,18,C.ink,'left',true);
    for(const v of [lo,0,hi]){line([[x,fy(v)],[x+w,fy(v)]],C.rule,.8);text(numeric(v),x-12,fy(v)+4,13,C.ink,'right')}
    const pts=Array.from(h,(v,j)=>[x+j/h.length*w,fy(v)]);pts.push([x+w,fy(h[0])]);
    ctx.beginPath();ctx.moveTo(x,fy(0));pts.forEach(p=>ctx.lineTo(...p));ctx.lineTo(x+w,fy(0));ctx.closePath();ctx.fillStyle='#176cc414';ctx.fill();line(pts,C.blue,2.8);
    for(const a of [-180,0,180])text(String(a),x+(a+180)/360*w,y+hh+25,14,C.ink,'center');text('Preferred direction (°)',x+w/2,y+hh+51,14,C.ink,'center');
  }
  function gridBump(d) {
    const h=row(d,'activity'),n=64,cx=290,cy=265,scale=137;
    const xy=(u,v)=>[cx+(u-v/2)*scale,cy-v*Math.sqrt(3)/2*scale];
    // Fixed neuron locations: only their activity color changes through time.
    for(let i=0;i<n;i++)for(let j=0;j<n;j++){const p=xy(-1+2*i/n,-1+2*j/n);dot(p[0],p[1],2.25,color(h[i*n+j]))}
    line([xy(-1,-1),xy(1,-1),xy(1,1),xy(-1,1),xy(-1,-1)],'#bacbde',1);
    text('θₓ',481,405,17);text('θᵧ',100,147,17);
    for(let i=0;i<252;i++){ctx.fillStyle=color(-1+2*i/251);ctx.fillRect(163+i,416,1.2,10)}
    text('≤ −1',163,447,12);text('0',289,447,12,C.ink,'center');text('≥ 1',415,447,12,C.ink,'right');
    if(mode==='recovery'){
      const xs=new Float32Array(n),ys=new Float32Array(n);for(let i=0;i<n;i++)for(let j=0;j<n;j++){xs[i]+=h[i*n+j]/n;ys[j]+=h[i*n+j]/n}
      const x=660,y=166,w=475,hh=190,lo=d.marginalRange[0],hi=d.marginalRange[1],fy=v=>y+hh-(v-lo)/(hi-lo)*hh;
      text('Phase-averaged activity',x,y-42,18,C.ink,'left',true);
      for(let j=0;j<3;j++){const v=lo+(hi-lo)*j/2;line([[x,fy(v)],[x+w,fy(v)]],C.rule,.8);text(numeric(v),x-12,fy(v)+4,13,C.ink,'right')}
      [xs,ys].forEach((s,j)=>{const points=Array.from(s,(v,k)=>[x+k/n*w,fy(v)]);points.push([x+w,fy(s[0])]);line(points,j?C.coral:C.blue,2.8)});
      line([[970,124],[995,124]],C.blue,3);text('θₓ',1003,129,15);line([[1060,124],[1085,124]],C.coral,3);text('θᵧ',1093,129,15);
      for(const a of [-180,0,180])text(String(a),x+(a+180)/360*w,y+hh+26,14,C.ink,'center');text('Preferred phase (°)',x+w/2,y+hh+52,14,C.ink,'center');
    }
  }
  function motion(d) {
    if(kind==='hd'){
      // Heading is conveyed by target/decoded traces, rather than a cloud rotating around a ring.
      text('Angular velocity (rad/s)',85,493,17,C.ink,'left',true);chart(d,d.velocities,95,528,440,90,{zero:true});
      text('Target and decoded heading (°)',698,493,17,C.ink,'left',true);chart(d,d.headings,705,528,425,90,{labels:['Decoded','Target']});
    }else{
      const pos=d.arrays.position,pred=d.path,box=d.pathBounds,x=674,y=154,w=455,h=265,scale=Math.min(w/(box[1]-box[0]),h/(box[3]-box[2]));
      const fx=(xx,yy)=>[x+w/2+(xx-(box[0]+box[1])/2)*scale,y+h/2-(yy-(box[2]+box[3])/2)*scale];
      const truth=[],learned=[];for(let i=0;i<d.arrays.time.length;i++){truth.push(fx(pos[i*2],pos[i*2+1]));learned.push(fx(pred[i*2],pred[i*2+1]))}
      text('Target and decoded position',665,115,18,C.ink,'left',true);line(truth,'#dbe4ef',1.5);line(truth.slice(0,frame(d)+1),C.coral,4,[6,4]);line(learned.slice(0,frame(d)+1),C.blue,2.3);dot(...learned[frame(d)],4.6,C.blue);
      dot(...truth[0],3.5,C.ink);text('Start',truth[0][0]+9,truth[0][1]-8,12);
      text('x (m)',1136,443,14,C.ink,'right');text('y (m)',651,157,14);
      line([[831,448],[858,448]],C.coral,3,[5,4]);text('Target',866,453,12);line([[973,448],[1000,448]],C.blue,2.3);text('Decoded',1008,453,12);
      text('Linear velocity (m/s)',85,493,17,C.ink,'left',true);chart(d,d.velocities,95,528,440,90,{labels:['vₓ','vᵧ'],zero:true});
      text('Wrapped phase error (°)',698,493,17,C.ink,'left',true);chart(d,d.errors,705,528,425,90,{labels:['θₓ','θᵧ'],zero:true});
    }
  }
  function draw() {
    if(!selected.length)return;ctx.clearRect(0,0,1200,680);ctx.fillStyle='white';ctx.fillRect(0,0,1200,680);
    if(mode==='manifold'){
      line([[600,43],[600,620]],C.rule,1);cloudView(selected[0],0,600);cloudView(selected[1],600,600);
      $('#clock').textContent=`HD ${selected[0].arrays.time[frame(selected[0])].toFixed(1)} s · Grid ${selected[1].arrays.time[frame(selected[1])].toFixed(1)} s`;
    }else{
      const d=selected[0];text(kind==='hd'?'Head-direction population':'Grid-cell population',54,65,23,C.ink,'left',true);text(mode==='recovery'?'Normal restoration':'Velocity-driven bump shifting',1140,65,16,C.ink,'right');
      kind==='hd'?hdBump(d):gridBump(d);line([[60,469],[1140,469]],C.rule,1);
      if(mode==='recovery'){
        text('Mean population energy',85,493,17,C.ink,'left',true);chart(d,[d.meanEnergy],95,528,440,90);
        text('Mean mapping distance',698,493,17,C.ink,'left',true);chart(d,[d.meanDistance],705,528,425,90,{zero:true});
      }else motion(d);
      $('#clock').textContent=d.arrays.time[frame(d)].toFixed(2)+' s';
    }
    $('#scrub').value=Math.round(progress*1000);
    const narrow=canvas.parentElement.clientWidth<700;
    const width=narrow?600:1200,height=narrow?(mode==='manifold'?1280:1270):680;
    if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height}
    display.clearRect(0,0,width,height);
    if(!narrow)display.drawImage(scene,0,0);
    else if(mode==='manifold'){
      display.drawImage(scene,0,0,600,640,0,0,600,640);
      display.drawImage(scene,600,0,600,640,0,640,600,640);
    }else{
      display.drawImage(scene,0,0,600,469,0,0,600,469);
      display.drawImage(scene,600,90,600,379,0,469,600,379);
      display.drawImage(scene,0,469,600,211,0,848,600,211);
      display.drawImage(scene,600,469,600,211,0,1059,600,211);
    }
  }
  function unwrap(a,dim,j) {const out=new Float32Array(a.length/dim);out[0]=a[j];for(let i=1;i<out.length;i++){const delta=a[i*dim+j]-a[(i-1)*dim+j];out[i]=out[i-1]+Math.atan2(Math.sin(delta),Math.cos(delta))}return out}
  async function getData(name) {
    if(cache.has(name))return cache.get(name);
    const meta=await fetch(`assets/data/${name}.json?v=3`).then(r=>{if(!r.ok)throw Error(r.status);return r.json()});
    const raw=await fetch(`assets/data/${meta.binary}?v=3`).then(r=>{if(!r.ok)throw Error(r.status);return r.arrayBuffer()});
    const arrays={};for(const[k,a]of Object.entries(meta.arrays))arrays[k]=new Float32Array(raw,a.offset*4,a.shape.reduce((p,v)=>p*v,1));
    const d={meta,arrays},n=meta.arrays.activity.shape[1];
    let amin=0,amax=1;for(const v of arrays.activity){amin=Math.min(amin,v);amax=Math.max(amax,v)}d.activityRange=[Math.floor(amin*10)/10,Math.ceil(amax*10)/10];
    if(meta.mode==='recovery'){
      d.bounds=referenceBounds(d);const b=d.bounds;let pr=0;
      for(let i=0;i<arrays.cloud.length;i+=meta.arrays.cloud.shape[2]){const x=arrays.cloud[i]-b.center[0],y=arrays.cloud[i+1]-b.center[1],z=arrays.cloud[i+2]-b.center[2]||0;pr=Math.max(pr,Math.abs(x*Math.cos(-.72)-y*Math.sin(-.72)),Math.abs((x*Math.sin(-.72)+y*Math.cos(-.72))*Math.sin(.96)+z*Math.cos(.96)))}b.projectedRange=2*pr;
      d.meanEnergy=average(d,'energy');d.meanDistance=average(d,'error',Math.sqrt(n));
      if(meta.kind==='grid'){
        let lo=0,hi=.15;for(let f=0;f<arrays.time.length;f++){const h=row(d,'activity',f);for(let j=0;j<64;j++){let a=0,b=0;for(let k=0;k<64;k++){a+=h[j*64+k]/64;b+=h[k*64+j]/64}lo=Math.min(lo,a,b);hi=Math.max(hi,a,b)}}d.marginalRange=[lo-.015,hi+.015];
      }
    }else{
      const dim=meta.kind==='hd'?1:2;d.velocities=Array.from({length:dim},(_,j)=>Array.from({length:arrays.time.length},(_,i)=>arrays.velocity[i*dim+j]));d.errors=Array.from({length:dim},(_,j)=>Array.from({length:arrays.time.length},(_,i)=>arrays.error[i*dim+j]));
      if(dim===1)d.headings=[unwrap(arrays.decoded,1,0),unwrap(arrays.target,1,0)].map(x=>Array.from(x,v=>v*180/Math.PI));
      else{const x=unwrap(arrays.decoded,2,0),y=unwrap(arrays.decoded,2,1);d.path=new Float32Array(x.length*2);let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;for(let i=0;i<x.length;i++){d.path[i*2]=x[i]/2;d.path[i*2+1]=(y[i]-.5*x[i])/Math.sqrt(3);xmin=Math.min(xmin,d.path[i*2],arrays.position[i*2]);xmax=Math.max(xmax,d.path[i*2],arrays.position[i*2]);ymin=Math.min(ymin,d.path[i*2+1],arrays.position[i*2+1]);ymax=Math.max(ymax,d.path[i*2+1],arrays.position[i*2+1])}d.pathBounds=[xmin-.1,xmax+.1,ymin-.1,ymax+.1]}
    }
    cache.set(name,d);return d;
  }
  function play(value) {playing=value;last=0;$('#play').textContent=value?'Ⅱ':'▶';$('#play').setAttribute('aria-label',value?'Pause trajectory':'Play trajectory')}
  async function load() {
    const id=++serial;play(false);selected=[];progress=0;elapsed=0;$('#loading').hidden=false;$('#loading').textContent='Loading trajectory…';['#play','#replay','#scrub'].forEach(s=>$(s).disabled=true);$('#population-selector').hidden=mode==='manifold';
    try {
      const items=await Promise.all((mode==='manifold'?['hd_recovery','grid_recovery']:[`${kind}_${mode}`]).map(getData));if(id!==serial)return;selected=items;
      const heading=mode==='manifold'?'Restoration toward head-direction rings and grid-cell tori.':mode==='recovery'?'Recovery of a localized activity bump.':'Bump shifting driven by angular or linear velocity.';$('#scene-title').textContent=heading;
      $('#scene-text').textContent=mode==='manifold'?`Each point represents one population state: ${items[0].meta.arrays.cloud.shape[1].toLocaleString()} HD states and ${items[1].meta.arrays.cloud.shape[1].toLocaleString()} Grid states. Both views use fixed linear readouts of the raw activity.`:mode==='recovery'?'Neurons stay at their preferred directions or phases; color shows their activity. The lower plots average energy and mapping distance over all 2,000 states.':kind==='hd'?'Angular velocity changes sign, shifting the bump in both directions. The lower plots show the velocity input and the target and decoded headings.':'Linear velocity moves the grid bump across the periodic phase lattice. The decoded position is compared with the target path, with velocity and phase error shown below.';
      $('#protocol-note').textContent=mode==='manifold'?'HD and Grid have different recovery durations. Playback aligns their progress; each clock shows its own simulated time.':mode==='recovery'?'Playback emphasizes the early recovery transient, then accelerates. Activity colors and plot scales remain fixed.':'Playback is accelerated. Velocity inputs and dynamics both update every 0.05 s.';
      $('#loading').hidden=true;['#play','#replay','#scrub'].forEach(s=>$(s).disabled=false);draw();
    }catch(e){if(id!==serial)return;$('#loading').textContent='The trajectory could not be loaded. Please refresh the page.';console.error(e)}
  }
  for(const b of document.querySelectorAll('[data-mode]'))b.addEventListener('click',()=>{mode=b.dataset.mode;for(const q of document.querySelectorAll('[data-mode]'))q.setAttribute('aria-pressed',String(q===b));load()});
  for(const b of document.querySelectorAll('[data-kind]'))b.addEventListener('click',()=>{kind=b.dataset.kind;for(const q of document.querySelectorAll('[data-kind]'))q.setAttribute('aria-pressed',String(q===b));load()});
  $('#play').addEventListener('click',()=>{if(!selected.length)return;if(progress>=1){progress=0;elapsed=0}play(!playing)});
  $('#replay').addEventListener('click',()=>{progress=0;elapsed=0;draw();play(true)});
  $('#scrub').addEventListener('input',e=>{if(!selected.length)return;play(false);progress=Number(e.target.value)/1000;elapsed=progress*16000;draw()});
  function animate(now){if(playing&&selected.length){if(last)elapsed+=(now-last)*Number($('#speed').value);progress=Math.min(1,elapsed/16000);draw();if(progress===1)play(false)}last=now;requestAnimationFrame(animate)}
  document.addEventListener('visibilitychange',()=>{if(document.hidden)play(false)});
  const film=$('#overview');if(!reduced)new IntersectionObserver(entries=>{for(const e of entries){if(e.isIntersecting)film.play().catch(()=>{});else film.pause()}},{threshold:.3}).observe(film);
  new ResizeObserver(()=>{if(selected.length)draw()}).observe(canvas.parentElement);
  load();requestAnimationFrame(animate);
})();
