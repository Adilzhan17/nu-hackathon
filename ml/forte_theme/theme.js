import {createMorph} from 'morphicons/dom';
const icons={home:'M3 10 12 3 21 10V21H15V14H9V21H3Z',analyze:'M4 4H15V20H4ZM15 9 21 6V18L15 15M8 8H11M8 12H11',training:'M4 20V14M10 20V9M16 20V4M3 4 8 7 13 3',monitor:'M3 12H7L10 4L14 20L17 12H21',review:'M4 3H16V21H4ZM8 8H12M8 12H12M15 14L17 16L22 10',objects:'M3 8V3H8M16 3H21V8M21 16V21H16M8 21H3V16M8 8H16V16H8Z',quality:'M12 3 21 7V12Q21 18 12 22Q3 18 3 12V7ZM8 12L11 15L16 9',menu:'M4 6H20M4 12H20M4 18H20',close:'M6 6L18 18M18 6L6 18',arrow:'M5 12H19M13 6L19 12L13 18'};
function svg(key){return `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${icons[key]}"></path></svg>`}
function init(){
 if(document.querySelector('.forte-sidebar'))return;
 if(window.self!==window.top){document.body.classList.add('forte-embedded');return}
 const panel=document.getElementById('live-resource-panel');const hero=document.querySelector('main>.hero');if(panel&&hero)panel.before(hero);
 const links=[['/analyze','Анализ видео','analyze'],['/training','Обучение моделей','training'],['/scenarios','Ресурсы и процессы','monitor'],['/label','Разметка действий','review'],['/label/objects','Разметка предметов','objects'],['/readiness','Качество моделей','quality']];
 document.querySelectorAll('header nav,main>nav').forEach(n=>n.classList.add('forte-legacy-nav'));
 document.querySelectorAll('header .logo').forEach(n=>n.classList.add('forte-legacy-nav'));
 const bar=document.createElement('aside');bar.className='forte-sidebar';bar.id='forte-navigation';
 bar.innerHTML=`<a class="forte-brand" href="/analyze" aria-label="Forte X — анализ видео"><img src="${window.FORTE_LOGO}" alt="Forte"><small>X · рабочая среда</small></a><nav aria-label="Основная навигация">${links.map(([url,label,icon],i)=>`${i===3?'<span class="forte-group-label">Работа с данными</span>':''}<a href="${url}" ${location.pathname===url?'aria-current="page"':''}>${svg(icon)}<span>${label}</span></a>`).join('')}</nav><div class="forte-sidebar-foot"><strong>Forte X</strong>Видеоконтроль кассовых операций</div>`;
 document.body.prepend(bar);
 bar.querySelectorAll('nav a').forEach((a,i)=>{const m=createMorph(a.querySelector('path'),icons[links[i][2]],{reducedMotion:'user'});a.addEventListener('pointerenter',()=>m.morphTo(icons.arrow,'snappy'));a.addEventListener('pointerleave',()=>m.morphTo(icons[links[i][2]],'snappy'));});
 const toggle=document.createElement('button');toggle.type='button';toggle.className='forte-menu';toggle.setAttribute('aria-controls',bar.id);toggle.setAttribute('aria-expanded','false');toggle.innerHTML=svg('menu')+'<span>Меню Forte</span>';document.body.prepend(toggle);const morph=createMorph(toggle.querySelector('path'),icons.menu,{reducedMotion:'user'});
 function close(){bar.classList.remove('is-open');toggle.setAttribute('aria-expanded','false');morph.morphTo(icons.menu)}
 toggle.onclick=()=>{const open=bar.classList.toggle('is-open');toggle.setAttribute('aria-expanded',String(open));morph.morphTo(icons[open?'close':'menu'])};document.addEventListener('keydown',e=>{if(e.key==='Escape')close()});document.addEventListener('click',e=>{if(!bar.contains(e.target)&&!toggle.contains(e.target))close()});
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();
