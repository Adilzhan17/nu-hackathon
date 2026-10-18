/* Review-time filtering only. Original scores and server reports remain immutable. */
(function(root){
'use strict';
function valid(w){return Number.isFinite(w.start_sec)&&Number.isFinite(w.end_sec)&&Number.isFinite(w.score)&&w.score>=0&&w.score<=1&&w.end_sec>w.start_sec;}
function partition(windows,duration,minVotes=1){
 if(!Number.isFinite(duration)||duration<=0)return [];
 const clean=windows.filter(valid),votesNeeded=Math.max(1,Math.floor(minVotes)||1);
 const clamp=t=>Math.max(0,Math.min(duration,t));
 const bounds=[...new Set([0,duration,...clean.flatMap(w=>[clamp(w.start_sec),clamp(w.end_sec)])])].sort((a,b)=>a-b);
 return bounds.slice(0,-1).map((start,i)=>{const end=bounds[i+1],mid=(start+end)/2,cover=clean.filter(w=>w.start_sec<=mid&&w.end_sec>mid),models=[...new Set(cover.filter(w=>w.suspect).map(w=>w.model_key||'single'))],observed=new Set(cover.map(w=>w.model_key||'single')).size;return {start_sec:start,end_sec:end,score:cover.length?Math.max(...cover.map(w=>w.score)):null,suspect:models.length>=votesNeeded,covered:observed>=votesNeeded,votes:models.length,models,observed_models:observed};});
}
function merge(parts){const events=[];for(const p of parts.filter(p=>p.suspect)){const prev=events.at(-1);if(prev&&Math.abs(prev.end_sec-p.start_sec)<1e-6){prev.end_sec=p.end_sec;prev.max_votes=Math.max(prev.max_votes,p.votes||1);if(p.score>prev.score){prev.score=p.score;prev.timecode_sec=(p.start_sec+p.end_sec)/2}}else events.push({start_sec:p.start_sec,end_sec:p.end_sec,score:p.score,timecode_sec:(p.start_sec+p.end_sec)/2,max_votes:p.votes||1});}return events;}
function events(windows,threshold){const sorted=windows.filter(valid).filter(w=>w.score>=threshold).map(w=>({...w,suspect:true})).sort((a,b)=>a.start_sec-b.start_sec||a.end_sec-b.end_sec),out=[];for(const w of sorted){const prev=out.at(-1);if(prev&&w.start_sec<=prev.end_sec){prev.end_sec=Math.max(prev.end_sec,w.end_sec);if(w.score>prev.score){prev.score=w.score;prev.timecode_sec=(w.start_sec+w.end_sec)/2}}else out.push({start_sec:w.start_sec,end_sec:w.end_sec,score:w.score,timecode_sec:(w.start_sec+w.end_sec)/2});}return out;}
function scoreAt(windows,t){const scores=windows.filter(valid).filter(w=>w.start_sec<=t&&w.end_sec>t).map(w=>w.score);return scores.length?Math.max(...scores):null;}
function review(job,threshold=.5,minVotes=1){
 const modelResults={},windows=[];
 for(const key of job.models||[]){const result=job.results?.[key]||{};const ws=(result.windows||[]).filter(valid);windows.push(...ws.map(w=>({...w,model_key:key,suspect:w.score>=threshold})));modelResults[key]={status:result.status||'queued',threshold,windows:ws.map(w=>({...w})),events:events(ws,threshold)};}
 const segments=partition(windows,job.duration_sec,minVotes);
 return {schema_version:1,report_type:'review_filter',source_job_id:job.id,source_updated_at:job.updated_at,source_report_url:'/api/uploads/'+job.id+'/report',settings:{threshold,min_models:minVotes,scope:'review_only',score_kind:'uncalibrated_model_score'},duration_sec:job.duration_sec,models:modelResults,segments,events:merge(segments)};
}
const api={partition,merge,events,scoreAt,review};if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.ForteTimeline=api;
})(typeof globalThis!=='undefined'?globalThis:this);
