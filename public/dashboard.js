let allChannels = [];

document.addEventListener("DOMContentLoaded", async () => {
  await loadChannels();
  await loadOverview();
});

function switchTab(name, btn) {
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(t => t.classList.remove("active"));
  btn.classList.add("active");
  document.getElementById("tab-" + name).classList.add("active");
}

async function loadChannels() {
  try {
    const r = await fetch("/api/channels");
    const d = await r.json();
    allChannels = d.channels;
    document.getElementById("channelCount").textContent = allChannels.length;
    populateDropdown();
  } catch (e) { console.error(e); }
}

function populateDropdown() {
  const f = document.getElementById("spacecraftFilter").value;
  const sel = document.getElementById("channelSelect");
  sel.innerHTML = "";
  const list = f === "all" ? allChannels : allChannels.filter(c => c.spacecraft === f);
  list.forEach(ch => {
    const o = document.createElement("option");
    o.value = ch.id;
    o.textContent = ch.id + " (" + ch.spacecraft + ") " + ch.anomalyPct + "%";
    sel.appendChild(o);
  });
}

document.getElementById("spacecraftFilter").addEventListener("change", populateDropdown);

function getConfig() {
  return {
    windowSize: +document.getElementById("windowSize").value,
    hiddenDim: +document.getElementById("hiddenDim").value,
    epochs: +document.getElementById("epochs").value,
    batchSize: +document.getElementById("batchSize").value,
    anomalyPercentile: +document.getElementById("anomalyPct").value
  };
}

function showStatus(m) { document.getElementById("statusBar").classList.remove("hidden"); document.getElementById("statusText").textContent = m; }
function hideStatus() { document.getElementById("statusBar").classList.add("hidden"); }

async function runChannel() {
  const cid = document.getElementById("channelSelect").value;
  const mt = document.getElementById("modelType").value;
  const btn = document.getElementById("runBtn");
  if (!cid) return;
  btn.disabled = true; btn.textContent = "Training...";
  showStatus("Training " + mt + " on " + cid + "... (1-3 min)");
  try {
    const r = await fetch("/api/channels/" + cid + "/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ modelType: mt, config: getConfig() })
    });
    if (!r.ok) throw new Error((await r.json()).error);
    const result = await r.json();
    renderResults(result);
  } catch (e) { alert("Error: " + e.message); }
  finally { btn.disabled = false; btn.textContent = "🚀 Run Detection"; hideStatus(); }
}

async function previewData() {
  const cid = document.getElementById("channelSelect").value;
  if (!cid) return;
  showStatus("Loading " + cid + "...");
  try {
    const r = await fetch("/api/channels/" + cid + "/data");
    const d = await r.json();
    document.getElementById("placeholder").classList.add("hidden");
    document.getElementById("results").classList.remove("hidden");
    const n = Math.min(d.testData.length, 2000);
    const t = Array.from({length:n},(_,i)=>i);
    const traces = [];
    for (let f = 0; f < Math.min(3, d.testData[0].length); f++)
      traces.push({x:t, y:d.testData.slice(0,n).map(r=>r[f]), mode:"lines", name:"F"+f, line:{width:1}});
    traces.push({x:t, y:d.testData.slice(0,n).map((r,i)=>d.labels[i]?r[0]:null), mode:"lines", name:"Anomaly", line:{color:"red",width:2}, fill:"tozeroy", fillcolor:"rgba(255,0,0,0.1)"});
    Plotly.newPlot("mainChart", traces, {title:"Preview: "+cid, template:"plotly_dark", paper_bgcolor:"#1a2332", plot_bgcolor:"#1a2332", height:500}, {responsive:true});
    document.getElementById("metricsGrid").innerHTML = "<div class='metric-card'><div class='label'>Channel</div><div class='value'>"+cid+"</div></div><div class='metric-card'><div class='label'>Spacecraft</div><div class='value'>"+d.spacecraft+"</div></div><div class='metric-card'><div class='label'>Points</div><div class='value'>"+d.testData.length+"</div></div>";
  } catch(e) { alert(e.message); }
  hideStatus();
}

function renderResults(res) {
  document.getElementById("placeholder").classList.add("hidden");
  document.getElementById("results").classList.remove("hidden");
  const pm = res.pointMetrics, rm = res.rangeMetrics;
  const gc = v => v > 0.7 ? "green" : v > 0.4 ? "yellow" : "red";
  document.getElementById("metricsGrid").innerHTML =
    ["Point P|"+pm.precision,"Point R|"+pm.recall,"Point F1|"+pm.f1,"Range P|"+rm.precision,"Range R|"+rm.recall,"Range F1|"+rm.f1]
    .map(s => { const [l,v] = s.split("|"); return "<div class='metric-card'><div class='label'>"+l+"</div><div class='value "+(l.includes("F1")?gc(+v):"")+"'>"+v+"</div></div>"; }).join("");

  fetch("/api/channels/" + res.chanId + "/data").then(r=>r.json()).then(d => {
    const n = Math.min(d.testData.length, res.testErrors.length, res.labels.length, res.predictions.length);
    const t = Array.from({length:n},(_,i)=>i);
    Plotly.newPlot("mainChart", [
      {x:t,y:d.testData.slice(0,n).map(r=>r[0]),mode:"lines",name:"Telemetry",line:{color:"steelblue",width:1},xaxis:"x",yaxis:"y"},
      {x:t,y:d.testData.slice(0,n).map((r,i)=>res.labels[i]?r[0]:null),mode:"lines",name:"True",line:{color:"red",width:2},fill:"tozeroy",fillcolor:"rgba(255,0,0,0.1)",xaxis:"x",yaxis:"y"},
      {x:t,y:res.testErrors.slice(0,n),mode:"lines",name:"Error",line:{color:"darkorange",width:1},xaxis:"x2",yaxis:"y2"},
      {x:[0,n],y:[res.threshold,res.threshold],mode:"lines",name:"Threshold",line:{color:"red",dash:"dash"},xaxis:"x2",yaxis:"y2"},
      {x:t,y:res.labels.slice(0,n).map(v=>v?0.9:null),mode:"lines",name:"True",fill:"tozeroy",fillcolor:"rgba(255,0,0,0.3)",line:{width:0},xaxis:"x3",yaxis:"y3"},
      {x:t,y:res.predictions.slice(0,n).map(v=>v?0.8:null),mode:"lines",name:"Pred",fill:"tozeroy",fillcolor:"rgba(0,100,255,0.25)",line:{width:0},xaxis:"x3",yaxis:"y3"}
    ], {title:res.chanId+" ("+res.modelType+")",template:"plotly_dark",paper_bgcolor:"#1a2332",plot_bgcolor:"#1a2332",height:600,
      xaxis:{domain:[0,1],anchor:"y"},yaxis:{domain:[0.68,1],title:"Value"},
      xaxis2:{domain:[0,1],anchor:"y2",matches:"x"},yaxis2:{domain:[0.36,0.62],title:"MSE"},
      xaxis3:{domain:[0,1],anchor:"y3",matches:"x",title:"Timestep"},yaxis3:{domain:[0,0.3],title:"Detect",range:[-0.1,1.1]},
      legend:{orientation:"h",y:1.05,x:0.5,xanchor:"center"},margin:{l:60,r:30,t:60,b:40}
    }, {responsive:true});
  });

  Plotly.newPlot("lossChart", [
    {x:res.trainLosses.map((_,i)=>i+1),y:res.trainLosses,mode:"lines",name:"Train",line:{color:"#3b82f6"}},
    {x:res.valLosses.map((_,i)=>i+1),y:res.valLosses,mode:"lines",name:"Val",line:{color:"#f59e0b"}}
  ], {title:"Loss",template:"plotly_dark",paper_bgcolor:"#1a2332",plot_bgcolor:"#1a2332",height:320,xaxis:{title:"Epoch"},yaxis:{title:"MSE"}}, {responsive:true});

  Plotly.newPlot("distChart", [
    {x:res.trainErrors,type:"histogram",name:"Train",opacity:0.6,marker:{color:"green"},nbinsx:80},
    {x:res.testErrors,type:"histogram",name:"Test",opacity:0.6,marker:{color:"orange"},nbinsx:80}
  ], {title:"Error Distribution",template:"plotly_dark",paper_bgcolor:"#1a2332",plot_bgcolor:"#1a2332",height:320,barmode:"overlay",xaxis:{title:"MSE"},yaxis:{title:"Count"},
    shapes:[{type:"line",x0:res.threshold,x1:res.threshold,y0:0,y1:1,yref:"paper",line:{color:"red",dash:"dash",width:2}}]
  }, {responsive:true});

  updateComparison(res);
}

function updateComparison(res) {
  const tbody = document.getElementById("compBody");
  const paperF1 = {"P-1":0.87,"P-2":0.91,"E-1":0.88,"A-1":0.90,"D-1":0.86,"G-1":0.84,"T-1":0.86,"F-1":0.88,"C-1":0.87,"M-1":0.81,"MSL-1":0.84};
  const pf = paperF1[res.chanId];
  const delta = pf ? (res.rangeMetrics.f1 - pf).toFixed(3) : "--";
  const cls = pf ? (delta > 0 ? "delta-pos" : "delta-neg") : "";
  const row = "<tr><td>"+res.chanId+"</td><td>"+res.spacecraft+"</td><td>"+res.rangeMetrics.f1+"</td><td>"+(pf||"--")+"</td><td class='"+cls+"'>"+(delta>0?"+":"")+delta+"</td></tr>";
  if (tbody.querySelector("td[colspan]")) tbody.innerHTML = "";
  tbody.innerHTML += row;
}

async function loadOverview() {
  try {
    const r = await fetch("/api/channels");
    const d = await r.json();
    const ch = d.channels;
    const smap = ch.filter(c=>c.spacecraft==="SMAP").length;
    const msl = ch.filter(c=>c.spacecraft==="MSL").length;
    document.getElementById("overviewMetrics").innerHTML =
      "<div class='metric-card'><div class='label'>Total</div><div class='value'>"+ch.length+"</div></div>"+
      "<div class='metric-card'><div class='label'>SMAP</div><div class='value'>"+smap+"</div></div>"+
      "<div class='metric-card'><div class='label'>MSL</div><div class='value'>"+msl+"</div></div>"+
      "<div class='metric-card'><div class='label'>Features</div><div class='value'>25</div></div>";
    Plotly.newPlot("overviewChart", [{x:ch.map(c=>c.anomalyPct),type:"histogram",nbinsx:25,marker:{color:"#f59e0b",opacity:0.8}}],
      {title:"Anomaly Rate Distribution",template:"plotly_dark",paper_bgcolor:"#1a2332",plot_bgcolor:"#1a2332",height:350,xaxis:{title:"Anomaly %"},yaxis:{title:"Channels"}}, {responsive:true});
  } catch(e) { console.error(e); }
}