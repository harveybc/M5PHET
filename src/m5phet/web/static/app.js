/* Single-owner client. Provider text and filenames are always rendered as text. */
const $ = id => document.getElementById(id);
const names = {classification:'Clasificación',regression_forecasting:'Pronóstico',representation_unsupervised:'Regímenes',causal_inference:'Inferencia causal',reinforcement_learning:'Política RL'};
const readers = {text:'Texto',json:'JSON',csv:'Dataset CSV',typed_request:'Petición tipada'};
const outputNames = {typed_questions:'Clasificación',point_forecast:'Pronóstico puntual',marginal_quantiles:'Cuantiles',hierarchical_regimes:'Jerarquía',causal_effect:'Efecto causal',policy_action:'Acción propuesta'};
let catalog, current, chats=[], selected=[], busy=false, uploading=false, poll=null;
function icon(name) { const n=document.createElement('i'); n.dataset.lucide=name; return n; }
function el(tag, text, cls) {const n=document.createElement(tag); if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
function icons(){lucide.createIcons();}
function notify(text=''){ $('notice').textContent=text;$('notice').hidden=!text; }
async function api(path, options={}) {
  const opts={...options,headers:{...options.headers}};
  if(opts.body && !(opts.body instanceof FormData)){opts.headers['Content-Type']='application/json';opts.body=JSON.stringify(opts.body);}
  const response=await fetch('/api'+path,opts);
  if(response.status===401){if(!$('login-dialog').open)$('login-dialog').showModal();throw new Error('Se requiere acceso al espacio');}
  if(response.status===204)return null;
  const data=await response.json();
  if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));
  return data;
}
function sidebar(open){document.body.classList.toggle('sidebar-open',open);$('shade').hidden=!open;}
async function list(){chats=await api('/chats');renderList();}
function renderList(){
  const query=$('search').value.toLocaleLowerCase();$('chat-list').replaceChildren();$('chat-count').textContent=chats.length;
  for(const chat of chats.filter(c=>c.title.toLocaleLowerCase().includes(query))){
    const b=el('button',undefined,'chat-item'+(current?.id===chat.id?' active':''));b.append(icon('message-square'));
    const text=el('span',undefined,'chat-text');text.append(el('strong',chat.title),el('small',new Date(chat.updated).toLocaleDateString('es',{day:'numeric',month:'short'})));b.append(text);
    b.onclick=()=>openChat(chat.id).catch(e=>notify(e.message));$('chat-list').append(b);
  }
}
function saveDraft(){if(current){localStorage.setItem('m5phet-draft-'+current.id,$('prompt').value);localStorage.setItem('m5phet-files-'+current.id,JSON.stringify(selected));}}
async function openChat(id){
  saveDraft();if(poll)clearTimeout(poll);
  current=await api('/chats/'+id);localStorage.setItem('m5phet-current',id);
  try{selected=JSON.parse(localStorage.getItem('m5phet-files-'+id)||'[]').filter(fid=>current.files.some(f=>f.id===fid));}catch{selected=[];}
  $('prompt').value=localStorage.getItem('m5phet-draft-'+id)||'';
  sidebar(false);render();renderList();watch();
}
async function newChat(){const created=await api('/chats',{method:'POST',body:{}});await list();await openChat(created.id);$('prompt').focus();}
function render(){
  if(!current)return;
  const cfg=current.config;
  $('chat-title').textContent=current.title;$('family-label').textContent=names[cfg.family]||cfg.family;
  $('input-label').textContent=readers[cfg.input];$('provider-label').textContent=cfg.provider||'Sin motor';
  $('output-label').textContent=outputNames[cfg.output_kind]||cfg.output_kind;
  const provider=catalog.providers.find(p=>p.name===cfg.provider);
  $('model-state').textContent=provider ? (provider.capabilities.backend==='fixture'?'NON_MODEL_FIXTURE · Sin modelo':cfg.provider+(provider.capabilities.known_states?.length?'':' · Sin estado configurado')):'Proveedor no instalado';
  $('messages').replaceChildren();
  current.messages.forEach(renderMessage);
  busy=current.messages.some(m=>m.status==='RUNNING');$('send').disabled=busy||uploading;$('empty').hidden=current.messages.length>0;
  $('export').disabled=false;renderFiles();icons();
}
function renderMessage(m){
  const article=el('article',undefined,'message '+m.role);const head=el('div',undefined,'message-head');
  head.append(icon(m.role==='user'?'user-round':'flower-2'),el('strong',m.role==='user'?'Tú':'M5PHET'));
  if(m.role==='assistant'){const tag=el('span',m.status,'tag'+(['OK','RUNNING'].includes(m.status)?'':' error'));head.append(tag);}
  head.append(el('time',new Date(m.created).toLocaleTimeString('es',{hour:'2-digit',minute:'2-digit'})));article.append(head);
  const body=el('div',undefined,'message-body');
  if(m.role==='user')body.textContent=m.content;
  else if(m.status==='RUNNING'){const run=el('div','Ejecutando...','running');run.prepend(icon('loader-circle'));body.append(run);}
  else {
    const detail=m.detail||{}, result=detail.result;
    if(m.status!=='OK'||!result){body.append(el('p',m.content));}
    else if(detail.config?.presentation==='json'){body.append(el('pre',JSON.stringify(result,null,2)));}
    else {
      for(const [name,answer] of Object.entries(result.outputs||{})){
        const p=answer.payload||{};
        if(p.label){body.append(el('div',p.label,'result-label'));const probs=p.uncalibrated_probabilities||p.probabilities;
          if(probs){for(const [label,v] of Object.entries(probs)){
            const row=el('div',undefined,'prob-row'),track=el('div',undefined,'prob-track'),fill=el('div',undefined,'prob-fill');
            fill.style.width=Math.max(0,Math.min(100,v*100))+'%';track.append(fill);row.append(el('span',label),track,el('span',(v*100).toFixed(2)+'%','prob-value'));body.append(row);
          }body.append(el('span','Probabilidades no calibradas','tag warn'));}
        }else {body.append(el('strong',name));renderPayload(body,p);}
      }
    }
    const meta=el('div',undefined,'result-meta');meta.append(el('span',detail.config?.provider||''));
    if(detail.backend==='fixture')meta.append(el('span','NON_MODEL_FIXTURE','tag warn'));
    if(detail.elapsed_seconds!==undefined)meta.append(el('span',detail.elapsed_seconds.toFixed(3)+' s'));
    meta.append(el('span','LOCAL_UNGOVERNED'));body.append(meta);
    const details=el('details');details.append(el('summary','Petición, configuración y resultado'),el('pre',JSON.stringify(detail,null,2)));body.append(details);
    const actions=el('div',undefined,'result-actions');const copy=el('button','Copiar JSON');copy.prepend(icon('copy'));copy.onclick=async()=>{try{await navigator.clipboard.writeText(JSON.stringify(detail,null,2));copy.textContent='Copiado';}catch{notify('No se pudo acceder al portapapeles');}};actions.append(copy);body.append(actions);
  }
  article.append(body);$('messages').append(article);
}
function renderPayload(parent,payload){
  const rows=payload.rows;
  if(Array.isArray(rows)&&rows.length&&typeof rows[0]==='object'){
    const box=el('div',undefined,'table-scroll'),table=el('table'),tr=el('tr'),keys=Object.keys(rows[0]);
    keys.forEach(k=>tr.append(el('th',k)));table.append(tr);
    rows.forEach(row=>{const tr=el('tr');keys.forEach(k=>tr.append(el('td',typeof row[k]==='object'?JSON.stringify(row[k]):String(row[k]))));table.append(tr);});box.append(table);parent.append(box);
  }else parent.append(el('pre',JSON.stringify(payload,null,2)));
}
function renderFiles(){
  $('attachments').replaceChildren();
  for(const f of current.files){
    const selectedNow=selected.includes(f.id),b=el('div',undefined,'attachment');
    const choose=el('button');choose.title=selectedNow?'Retirar de esta pregunta':'Incluir en esta pregunta';choose.setAttribute('aria-label',choose.title+' '+f.name);choose.append(icon(selectedNow?'check-circle-2':'circle'));choose.onclick=()=>{selected=selectedNow?selected.filter(id=>id!==f.id):[...selected,f.id];saveDraft();renderFiles();icons();};
    const a=el('a',f.name);a.href=`/api/chats/${current.id}/files/${f.id}`;a.download=f.name;a.title=f.sha256;b.append(choose,a);$('attachments').append(b);
  }
}
async function watch(){
  if(!current||!busy)return;
  const id=current.id;
  poll=setTimeout(async()=>{try{const updated=await api('/chats/'+id);if(current?.id===id){current=updated;render();$('messages').scrollTop=$('messages').scrollHeight;watch();}}catch(e){notify(e.message);}},650);
}
$('composer').onsubmit=async e=>{
  e.preventDefault();const prompt=$('prompt').value.trim();if(!prompt||busy||uploading)return;
  notify();$('send').disabled=true;
  try{
    const id=current.id;
    await api(`/chats/${id}/messages`,{method:'POST',body:{prompt,file_ids:selected,client_id:crypto.randomUUID()}});
    if(current.title==='Nuevo chat')await api('/chats/'+id,{method:'PATCH',body:{title:prompt.slice(0,80)}});
    $('prompt').value='';localStorage.removeItem('m5phet-draft-'+id);current=await api('/chats/'+id);render();await list();$('messages').scrollTop=$('messages').scrollHeight;watch();
  }catch(error){notify(error.message);$('send').disabled=false;}
};
$('prompt').oninput=()=>{saveDraft();$('prompt').style.height='auto';$('prompt').style.height=Math.min(180,$('prompt').scrollHeight)+'px';};
$('prompt').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();$('composer').requestSubmit();}};
$('attach').onclick=()=>$('file-input').click();
$('file-input').onchange=async()=>{
  if(!current)return;const cid=current.id;uploading=true;$('send').disabled=true;notify();
  try{for(const file of $('file-input').files){const form=new FormData();form.append('file',file);const saved=await api(`/chats/${cid}/files`,{method:'POST',body:form});if(current?.id===cid)selected.push(saved.id);}
    if(current?.id===cid){current=await api('/chats/'+cid);saveDraft();render();}}
  catch(e){notify(e.message);}finally{uploading=false;$('send').disabled=busy;$('file-input').value='';}
};
function providerOptions(){
  const select=$('cfg-provider');select.replaceChildren();
  for(const p of catalog.providers){const option=el('option',p.name+(p.capabilities.backend==='fixture'?' (sin modelo)':''));option.value=p.name;select.append(option);}
  if(!catalog.providers.some(p=>p.name===current.config.provider)){const o=el('option',current.config.provider+' (no instalado)');o.value=current.config.provider;select.append(o);}
  select.value=current.config.provider;contractOptions();
}
function contractOptions(){
  const p=catalog.providers.find(p=>p.name===$('cfg-provider').value);const select=$('cfg-contract');select.replaceChildren();
  for(const c of p?.capabilities.supported||[]){if(c.operation!=='infer')continue;const o=el('option',(names[c.family]||c.family)+' / '+(outputNames[c.output_kind]||c.output_kind));o.value=c.family+'|'+c.output_kind;select.append(o);}
  if(!select.options.length){const o=el('option','Contrato no disponible');o.value=current.config.family+'|'+current.config.output_kind;select.append(o);}
  const selectedValue=current.config.family+'|'+current.config.output_kind;
  if([...select.options].some(o=>o.value===selectedValue))select.value=selectedValue;
  const state=$('cfg-state');state.replaceChildren();const automatic=el('option','Estado único del proveedor');automatic.value='';state.append(automatic);
  for(const name of p?.capabilities.known_states||[]){const o=el('option',name);o.value=name;state.append(o);}
  state.value=current.config.state||'';
  $('provider-info').textContent=p?JSON.stringify({backend:p.capabilities.backend||'plugin',device:p.capabilities.device||'declarado por el proveedor',states:p.capabilities.known_states?.length||0}):'Proveedor no instalado';
}
function optionRow(label='',description=''){
  const row=el('div',undefined,'option-row');const a=el('input'),b=el('input'),remove=el('button');a.value=label;a.placeholder='Etiqueta';a.setAttribute('aria-label','Etiqueta');b.value=description;b.placeholder='Descripción';b.setAttribute('aria-label','Descripción de opción');remove.type='button';remove.className='icon';remove.title='Quitar opción';remove.setAttribute('aria-label','Quitar opción');remove.append(icon('x'));remove.onclick=()=>row.remove();row.append(a,b,remove);$('option-list').append(row);
}
function settings(){
  if(!current)return;const c=current.config;$('cfg-title').value=current.title;
  for(const key of ['input','context','asset','language','presentation'])$('cfg-'+key).value=c[key];
  $('cfg-asof').value=c.as_of;$('cfg-age').value=c.max_age_seconds;$('cfg-parameters').value=JSON.stringify(c.parameters||{},null,2);
  $('option-list').replaceChildren();c.options.forEach(pair=>optionRow(...pair));providerOptions();$('config-error').textContent='';$('config-dialog').showModal();icons();
}
$('settings').onclick=settings;$('engine-settings').onclick=()=>{settings();tab('engine');};
function tab(name){document.querySelectorAll('[data-tab]').forEach(b=>b.setAttribute('aria-selected',b.dataset.tab===name));document.querySelectorAll('[data-pane]').forEach(s=>s.hidden=s.dataset.pane!==name);}
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>tab(b.dataset.tab));
document.querySelectorAll('.close-dialog').forEach(b=>b.onclick=()=>$('config-dialog').close());
$('cfg-provider').onchange=contractOptions;$('add-option').onclick=()=>{optionRow();icons();};
$('config-form').onsubmit=async e=>{
  e.preventDefault();try{const cfg={...current.config};for(const key of ['input','context','asset','language','presentation','provider','state'])cfg[key]=$('cfg-'+key).value;
    [cfg.family,cfg.output_kind]=$('cfg-contract').value.split('|');cfg.as_of=$('cfg-asof').value;cfg.max_age_seconds=Number($('cfg-age').value);cfg.parameters=JSON.parse($('cfg-parameters').value);
    cfg.options=[...$('option-list').children].map(row=>[...row.querySelectorAll('input')].map(i=>i.value));
    current=await api('/chats/'+current.id,{method:'PATCH',body:{title:$('cfg-title').value,config:cfg}});$('config-dialog').close();render();await list();
  }catch(error){$('config-error').textContent=error.message;}
};
$('delete-chat').onclick=()=>$('confirm-dialog').showModal();$('cancel-delete').onclick=()=>$('confirm-dialog').close();
$('confirm-delete').onclick=async()=>{try{const id=current.id;await api('/chats/'+id,{method:'DELETE'});localStorage.removeItem('m5phet-draft-'+id);current=null;$('confirm-dialog').close();$('config-dialog').close();await list();if(chats.length)await openChat(chats[0].id);else await newChat();}catch(e){$('confirm-dialog').close();$('config-error').textContent=e.message;}};
$('new-chat').onclick=()=>newChat().catch(e=>notify(e.message));$('search').oninput=renderList;
$('menu').onclick=()=>sidebar(true);$('shade').onclick=()=>sidebar(false);
$('export').onclick=()=>{if(current)window.location.href='/api/chats/'+current.id+'/export';};
$('login-form').onsubmit=async e=>{e.preventDefault();try{await api('/login',{method:'POST',body:{token:$('access-token').value}});$('access-token').value='';$('login-dialog').close();await boot();}catch(error){$('login-error').textContent=error.message;}};
async function boot(){
  catalog=await api('/catalog');await list();const id=localStorage.getItem('m5phet-current');
  if(chats.length)await openChat(chats.some(c=>c.id===id)?id:chats[0].id);else await newChat();
  $('examples').replaceChildren();
  for(const example of catalog.examples||[]){const b=el('button',example.title);b.prepend(icon('flask-conical'));b.onclick=async()=>{try{
      if(current.messages.length)await newChat();
      current=await api('/chats/'+current.id,{method:'PATCH',body:{title:example.title,config:{...catalog.defaults,...example.config}}});
      const file=new Blob([typeof example.data==='string'?example.data:JSON.stringify(example.data)],{type:'application/json'}),form=new FormData();form.append('file',file,typeof example.data==='string'?'context.txt':'data.json');
      const saved=await api(`/chats/${current.id}/files`,{method:'POST',body:form});selected=[saved.id];current=await api('/chats/'+current.id);$('prompt').value=example.prompt;saveDraft();render();await list();$('prompt').focus();
    }catch(e){notify(e.message);}};$('examples').append(b);}
  icons();
}
boot().catch(e=>notify(e.message));
