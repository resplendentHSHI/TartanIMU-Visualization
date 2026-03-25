#!/usr/bin/env python3
"""
Real-time 3D Foxglove-style visualizer for NPZ alignment data.

Loads NPZ files produced by alignment_original.py and serves an interactive
web-based 3D visualization using Three.js. Uses EgoVerse pose_utils for
quaternion handling.

Usage:
    python visualize_live.py <npz_file> [--port 8888]
    python visualize_live.py Helmet_Poser_Aligned/A_c/A_c.npz
    python visualize_live.py debug_dataset/car/0/0.npz --port 9000
"""
import argparse
import http.server
import json
import sys
import threading
import webbrowser
from pathlib import Path

import numpy as np

# EgoVerse pose utilities
sys.path.insert(0, str(Path(__file__).resolve().parent / "EgoVerse"))
from egomimic.utils.pose_utils import xyzw_to_wxyz, _xyzwxyz_to_matrix, _matrix_to_xyzypr


def load_and_prepare(npz_path, max_points=8000):
    """Load NPZ and prepare JSON-serializable data for the frontend."""
    data = np.load(npz_path, allow_pickle=True)
    ts = data["retargetted_ts"]
    imu = data["retargetted_imu"]   # Nx6
    pos = data["retargetted_pos"]   # Nx3
    quat = data["retargetted_quat"] # Nx4 XYZW

    # Make timestamps relative
    ts = ts - ts[0]

    # Subsample for browser performance
    n = len(ts)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=int)
        ts, imu, pos, quat = ts[idx], imu[idx], pos[idx], quat[idx]

    # Quaternion -> Euler via EgoVerse pose_utils
    wxyz = xyzw_to_wxyz(quat)
    xyzwxyz = np.hstack([np.zeros((len(wxyz), 3)), wxyz])
    mats = _xyzwxyz_to_matrix(xyzwxyz)
    euler = np.degrees(_matrix_to_xyzypr(mats)[:, 3:])  # yaw, pitch, roll

    # Accel magnitude
    accel_mag = np.linalg.norm(imu[:, :3], axis=1)

    return {
        "name": Path(npz_path).stem,
        "count": len(ts),
        "duration": float(ts[-1]),
        "ts": ts.tolist(),
        "pos": pos.tolist(),
        "quat_xyzw": quat.tolist(),
        "imu": imu.tolist(),
        "euler": euler.tolist(),
        "accel_mag": accel_mag.tolist(),
    }


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>IMU Visualizer</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#1a1a2e; color:#e0e0e0; font-family:'Consolas','SF Mono',monospace; overflow:hidden; }

#app { display:grid; grid-template-columns:1fr 340px; grid-template-rows:40px 1fr 54px; height:100vh; }

/* Top bar */
#topbar { grid-column:1/3; background:#16213e; display:flex; align-items:center; padding:0 16px; gap:16px; border-bottom:1px solid #0f3460; }
#topbar h1 { font-size:13px; color:#e94560; font-weight:700; letter-spacing:1px; }
#topbar .info { font-size:11px; color:#888; }

/* 3D viewport */
#viewport { position:relative; overflow:hidden; background:#0a0a1a; }
#viewport canvas { display:block; }
#view-controls { position:absolute; top:10px; left:10px; display:flex; gap:6px; }
#view-controls button { background:#16213e; border:1px solid #0f3460; color:#e0e0e0; padding:4px 10px; font-size:11px; cursor:pointer; border-radius:3px; font-family:inherit; }
#view-controls button:hover, #view-controls button.active { background:#0f3460; color:#e94560; }
#frame-info { position:absolute; bottom:10px; left:10px; font-size:11px; color:#888; background:rgba(10,10,26,0.85); padding:6px 10px; border-radius:4px; }

/* Side panel */
#sidepanel { background:#16213e; border-left:1px solid #0f3460; overflow-y:auto; padding:8px; display:flex; flex-direction:column; gap:6px; }
.panel-section { background:#1a1a2e; border:1px solid #0f3460; border-radius:4px; padding:8px; }
.panel-section h3 { font-size:11px; color:#e94560; margin-bottom:6px; text-transform:uppercase; letter-spacing:1px; }
.panel-section canvas { width:100%; height:80px; display:block; border-radius:2px; }
.stat-row { display:flex; justify-content:space-between; font-size:11px; padding:2px 0; }
.stat-label { color:#888; }
.stat-value { color:#4fc3f7; font-weight:600; }

/* Timeline */
#timeline { grid-column:1/3; background:#16213e; border-top:1px solid #0f3460; display:flex; align-items:center; padding:0 16px; gap:12px; }
#timeline button { background:none; border:1px solid #0f3460; color:#e0e0e0; width:32px; height:28px; cursor:pointer; border-radius:3px; font-size:14px; font-family:inherit; }
#timeline button:hover { background:#0f3460; color:#e94560; }
#timeline button.active { background:#e94560; color:#fff; border-color:#e94560; }
#scrubber { flex:1; -webkit-appearance:none; appearance:none; height:4px; background:#0f3460; border-radius:2px; outline:none; }
#scrubber::-webkit-slider-thumb { -webkit-appearance:none; width:14px; height:14px; background:#e94560; border-radius:50%; cursor:pointer; }
#scrubber::-moz-range-thumb { width:14px; height:14px; background:#e94560; border-radius:50%; cursor:pointer; border:none; }
#time-display { font-size:12px; color:#4fc3f7; min-width:120px; text-align:right; }
#speed-select { background:#1a1a2e; border:1px solid #0f3460; color:#e0e0e0; padding:2px 6px; font-size:11px; border-radius:3px; font-family:inherit; }
</style>
</head>
<body>
<div id="app">
  <div id="topbar">
    <h1>&#9670; IMU VISUALIZER</h1>
    <span class="info" id="dataset-info"></span>
  </div>
  <div id="viewport">
    <div id="view-controls">
      <button class="active" data-view="perspective">3D</button>
      <button data-view="top">Top</button>
      <button data-view="side">Side</button>
      <button data-view="front">Front</button>
    </div>
    <div id="frame-info"></div>
  </div>
  <div id="sidepanel">
    <div class="panel-section">
      <h3>Accelerometer</h3>
      <canvas id="chart-accel"></canvas>
    </div>
    <div class="panel-section">
      <h3>Gyroscope</h3>
      <canvas id="chart-gyro"></canvas>
    </div>
    <div class="panel-section">
      <h3>|Accel| Gravity Check</h3>
      <canvas id="chart-mag"></canvas>
    </div>
    <div class="panel-section">
      <h3>Orientation (Euler)</h3>
      <canvas id="chart-euler"></canvas>
    </div>
    <div class="panel-section">
      <h3>Current Frame</h3>
      <div id="stats"></div>
    </div>
  </div>
  <div id="timeline">
    <button id="btn-play" title="Play/Pause">&#9654;</button>
    <button id="btn-reset" title="Reset">&#9632;</button>
    <input type="range" id="scrubber" min="0" max="1000" value="0">
    <select id="speed-select">
      <option value="0.25">0.25x</option>
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
      <option value="5">5x</option>
      <option value="10">10x</option>
    </select>
    <span id="time-display">0.00s / 0.00s</span>
  </div>
</div>

<script type="importmap">
{ "imports": { "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js", "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/" } }
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Data ──
const D = /*__DATA__*/null;

const COUNT = D.count;
const TS = D.ts;
const POS = D.pos;
const QUAT = D.quat_xyzw;
const IMU = D.imu;
const EULER = D.euler;
const AMAG = D.accel_mag;
const DURATION = D.duration;

document.getElementById('dataset-info').textContent =
  `${D.name}  |  ${COUNT} samples  |  ${DURATION.toFixed(1)}s`;

// ── Three.js Setup ──
const viewport = document.getElementById('viewport');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x0a0a1a);
viewport.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 500);

// Compute scene center and scale
let posArr = POS.map(p => new THREE.Vector3(p[0], p[1], p[2]));
let bbox = new THREE.Box3().setFromPoints(posArr);
let center = new THREE.Vector3();
bbox.getCenter(center);
let size = bbox.getSize(new THREE.Vector3()).length();
let viewDist = Math.max(size * 1.5, 1.0);

camera.position.set(center.x + viewDist * 0.6, center.y + viewDist * 0.4, center.z + viewDist * 0.6);
camera.lookAt(center);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(center);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.update();

// ── Lighting ──
scene.add(new THREE.AmbientLight(0x404060, 2));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
dirLight.position.set(5, 10, 5);
scene.add(dirLight);

// ── Grid ──
const gridSize = Math.ceil(size * 2);
const grid = new THREE.GridHelper(gridSize, gridSize * 2, 0x0f3460, 0x0a0a2e);
grid.position.y = bbox.min.y - 0.01;
scene.add(grid);

// ── World Axes ──
const axLen = size * 0.3;
function makeAxis(dir, color) {
  const geo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0,0,0),
    new THREE.Vector3(dir[0]*axLen, dir[1]*axLen, dir[2]*axLen)
  ]);
  return new THREE.Line(geo, new THREE.LineBasicMaterial({ color }));
}
const worldAxes = new THREE.Group();
worldAxes.add(makeAxis([1,0,0], 0xe94560));
worldAxes.add(makeAxis([0,1,0], 0x4fc3f7));
worldAxes.add(makeAxis([0,0,1], 0x50c878));
worldAxes.position.set(bbox.min.x - size*0.1, bbox.min.y, bbox.min.z - size*0.1);
scene.add(worldAxes);

// ── Full Trajectory (dim trail) ──
const trailGeo = new THREE.BufferGeometry();
const trailPositions = new Float32Array(COUNT * 3);
for (let i = 0; i < COUNT; i++) {
  trailPositions[i*3]   = POS[i][0];
  trailPositions[i*3+1] = POS[i][1];
  trailPositions[i*3+2] = POS[i][2];
}
trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPositions, 3));
const trailLine = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0x0f3460, linewidth: 1 }));
scene.add(trailLine);

// ── Active Trail (bright, up to current frame) ──
const activeGeo = new THREE.BufferGeometry();
const activePositions = new Float32Array(COUNT * 3);
activeGeo.setAttribute('position', new THREE.BufferAttribute(activePositions, 3));
activeGeo.setDrawRange(0, 0);
const activeLine = new THREE.Line(activeGeo, new THREE.LineBasicMaterial({ color: 0xe94560, linewidth: 2 }));
scene.add(activeLine);

// ── Pose Frame (coordinate axes at current position) ──
const frameGroup = new THREE.Group();
const fLen = Math.max(size * 0.08, 0.05);
function makeFrameAxis(color) {
  const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(fLen,0,0)]);
  return new THREE.Line(geo, new THREE.LineBasicMaterial({ color, linewidth: 2 }));
}
const fAxisX = makeFrameAxis(0xe94560); // red = X
const fAxisY = makeFrameAxis(0x4fc3f7); // blue = Y
fAxisY.geometry.setFromPoints([new THREE.Vector3(), new THREE.Vector3(0,fLen,0)]);
const fAxisZ = makeFrameAxis(0x50c878); // green = Z
fAxisZ.geometry.setFromPoints([new THREE.Vector3(), new THREE.Vector3(0,0,fLen)]);
frameGroup.add(fAxisX, fAxisY, fAxisZ);
scene.add(frameGroup);

// ── Position Marker (sphere) ──
const markerGeo = new THREE.SphereGeometry(Math.max(size * 0.012, 0.008), 16, 16);
const markerMat = new THREE.MeshPhongMaterial({ color: 0xe94560, emissive: 0xe94560, emissiveIntensity: 0.5 });
const marker = new THREE.Mesh(markerGeo, markerMat);
scene.add(marker);

// ── Resize ──
function onResize() {
  const w = viewport.clientWidth, h = viewport.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
window.addEventListener('resize', onResize);
onResize();

// ── Playback State ──
let playing = false;
let currentTime = 0;
let speed = 1;
let currentIdx = 0;

const btnPlay = document.getElementById('btn-play');
const btnReset = document.getElementById('btn-reset');
const scrubber = document.getElementById('scrubber');
const timeDisplay = document.getElementById('time-display');
const speedSelect = document.getElementById('speed-select');

btnPlay.addEventListener('click', () => {
  playing = !playing;
  btnPlay.innerHTML = playing ? '&#10074;&#10074;' : '&#9654;';
  btnPlay.classList.toggle('active', playing);
});
btnReset.addEventListener('click', () => {
  currentTime = 0;
  playing = false;
  btnPlay.innerHTML = '&#9654;';
  btnPlay.classList.remove('active');
  updateFrame(0);
});
scrubber.addEventListener('input', () => {
  currentTime = (scrubber.value / 1000) * DURATION;
  updateFrame(currentTime);
});
speedSelect.addEventListener('change', () => { speed = parseFloat(speedSelect.value); });

// ── View Presets ──
document.querySelectorAll('#view-controls button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#view-controls button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const v = btn.dataset.view;
    const d = viewDist;
    if (v === 'perspective') {
      camera.position.set(center.x+d*0.6, center.y+d*0.4, center.z+d*0.6);
    } else if (v === 'top') {
      camera.position.set(center.x, center.y+d, center.z);
    } else if (v === 'side') {
      camera.position.set(center.x+d, center.y, center.z);
    } else if (v === 'front') {
      camera.position.set(center.x, center.y, center.z+d);
    }
    camera.lookAt(center);
    controls.target.copy(center);
    controls.update();
  });
});

// ── Side Panel Charts ──
const chartConfigs = [
  { id: 'chart-accel', channels: [0,1,2], colors: ['#e94560','#4fc3f7','#50c878'], labels: ['ax','ay','az'], src: 'imu' },
  { id: 'chart-gyro',  channels: [3,4,5], colors: ['#e94560','#4fc3f7','#50c878'], labels: ['gx','gy','gz'], src: 'imu' },
  { id: 'chart-mag',   channels: [0],     colors: ['#4fc3f7'], labels: ['|a|'], src: 'amag', refLine: 9.81 },
  { id: 'chart-euler', channels: [0,1,2], colors: ['#e94560','#4fc3f7','#50c878'], labels: ['yaw','pitch','roll'], src: 'euler' },
];

function drawChart(cfg, curIdx) {
  const canvas = document.getElementById(cfg.id);
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.clientWidth * window.devicePixelRatio;
  const H = canvas.height = canvas.clientHeight * window.devicePixelRatio;
  ctx.clearRect(0, 0, W, H);

  // Determine visible window (show ~4s around current time)
  const windowSec = 4;
  const tCenter = TS[curIdx];
  const tMin = tCenter - windowSec / 2;
  const tMax = tCenter + windowSec / 2;

  // Find data range for visible window
  let iStart = 0, iEnd = COUNT - 1;
  for (let i = 0; i < COUNT; i++) { if (TS[i] >= tMin) { iStart = i; break; } }
  for (let i = COUNT - 1; i >= 0; i--) { if (TS[i] <= tMax) { iEnd = i; break; } }

  // Compute y range
  let yMin = Infinity, yMax = -Infinity;
  for (let i = iStart; i <= iEnd; i++) {
    for (const ch of cfg.channels) {
      const v = cfg.src === 'imu' ? IMU[i][ch] : cfg.src === 'amag' ? AMAG[i] : EULER[i][ch];
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
    }
  }
  if (cfg.refLine !== undefined) { yMin = Math.min(yMin, cfg.refLine - 1); yMax = Math.max(yMax, cfg.refLine + 1); }
  const yPad = (yMax - yMin) * 0.1 || 1;
  yMin -= yPad; yMax += yPad;

  const xMap = t => ((t - tMin) / (tMax - tMin)) * W;
  const yMap = v => H - ((v - yMin) / (yMax - yMin)) * H;

  // Background
  ctx.fillStyle = '#0a0a1a';
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = '#1a1a2e';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = (i / 4) * H;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }

  // Reference line
  if (cfg.refLine !== undefined) {
    ctx.strokeStyle = '#e9456066';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    const ry = yMap(cfg.refLine);
    ctx.beginPath(); ctx.moveTo(0, ry); ctx.lineTo(W, ry); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#e94560';
    ctx.font = `${10 * window.devicePixelRatio}px monospace`;
    ctx.fillText(`g=${cfg.refLine}`, 4, ry - 4);
  }

  // Data lines
  const step = Math.max(1, Math.floor((iEnd - iStart) / (W / 2)));
  for (let ci = 0; ci < cfg.channels.length; ci++) {
    const ch = cfg.channels[ci];
    ctx.strokeStyle = cfg.colors[ci];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let first = true;
    for (let i = iStart; i <= iEnd; i += step) {
      const v = cfg.src === 'imu' ? IMU[i][ch] : cfg.src === 'amag' ? AMAG[i] : EULER[i][ch];
      const x = xMap(TS[i]), y = yMap(v);
      if (first) { ctx.moveTo(x, y); first = false; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Playhead
  const px = xMap(tCenter);
  ctx.strokeStyle = '#ffffff44';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke();

  // Legend
  ctx.font = `${9 * window.devicePixelRatio}px monospace`;
  for (let ci = 0; ci < cfg.labels.length; ci++) {
    ctx.fillStyle = cfg.colors[ci];
    ctx.fillText(cfg.labels[ci], W - 50 * window.devicePixelRatio, 14 * window.devicePixelRatio * (ci + 1));
  }
}

// ── Frame Update ──
function updateFrame(t) {
  // Binary search for nearest index
  let lo = 0, hi = COUNT - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (TS[mid] < t) lo = mid + 1; else hi = mid;
  }
  currentIdx = lo;
  const i = currentIdx;

  // Update marker
  marker.position.set(POS[i][0], POS[i][1], POS[i][2]);

  // Update pose frame
  const q = new THREE.Quaternion(QUAT[i][0], QUAT[i][1], QUAT[i][2], QUAT[i][3]);
  frameGroup.position.copy(marker.position);
  frameGroup.quaternion.copy(q);

  // Update active trail
  for (let j = 0; j <= i; j++) {
    activePositions[j*3]   = POS[j][0];
    activePositions[j*3+1] = POS[j][1];
    activePositions[j*3+2] = POS[j][2];
  }
  activeGeo.attributes.position.needsUpdate = true;
  activeGeo.setDrawRange(0, i + 1);

  // Scrubber
  scrubber.value = (t / DURATION) * 1000;
  timeDisplay.textContent = `${t.toFixed(2)}s / ${DURATION.toFixed(2)}s`;

  // Frame info
  document.getElementById('frame-info').textContent =
    `Frame ${i}/${COUNT-1}  |  ` +
    `Pos: (${POS[i][0].toFixed(3)}, ${POS[i][1].toFixed(3)}, ${POS[i][2].toFixed(3)})  |  ` +
    `|a|: ${AMAG[i].toFixed(2)} m/s²`;

  // Stats panel
  document.getElementById('stats').innerHTML =
    `<div class="stat-row"><span class="stat-label">Time</span><span class="stat-value">${TS[i].toFixed(3)}s</span></div>` +
    `<div class="stat-row"><span class="stat-label">Position</span><span class="stat-value">${POS[i].map(v=>v.toFixed(3)).join(', ')}</span></div>` +
    `<div class="stat-row"><span class="stat-label">Quat (xyzw)</span><span class="stat-value">${QUAT[i].map(v=>v.toFixed(3)).join(', ')}</span></div>` +
    `<div class="stat-row"><span class="stat-label">Euler (ypr)</span><span class="stat-value">${EULER[i].map(v=>v.toFixed(1)).join('°, ')}°</span></div>` +
    `<div class="stat-row"><span class="stat-label">Accel</span><span class="stat-value">${IMU[i].slice(0,3).map(v=>v.toFixed(2)).join(', ')}</span></div>` +
    `<div class="stat-row"><span class="stat-label">Gyro</span><span class="stat-value">${IMU[i].slice(3,6).map(v=>v.toFixed(3)).join(', ')}</span></div>` +
    `<div class="stat-row"><span class="stat-label">|Accel|</span><span class="stat-value">${AMAG[i].toFixed(3)} m/s²</span></div>`;

  // Charts
  for (const cfg of chartConfigs) drawChart(cfg, i);
}

// ── Animation Loop ──
let lastT = null;
function animate(timestamp) {
  requestAnimationFrame(animate);

  if (playing) {
    if (lastT !== null) {
      const dt = (timestamp - lastT) / 1000;
      currentTime += dt * speed;
      if (currentTime >= DURATION) {
        currentTime = 0; // loop
      }
    }
    updateFrame(currentTime);
  }
  lastT = timestamp;

  controls.update();
  renderer.render(scene, camera);
}

// Initial render
updateFrame(0);
requestAnimationFrame(animate);
</script>
</body>
</html>"""


def serve(npz_path, port):
    data = load_and_prepare(npz_path)
    data_json = json.dumps(data)
    html = HTML_PAGE.replace("/*__DATA__*/null", data_json)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(html.encode())

        def log_message(self, fmt, *args):
            pass  # silence request logs

    server = http.server.HTTPServer(("0.0.0.0", port), Handler)
    url = f"http://localhost:{port}"
    print(f"\n  ╔══════════════════════════════════════════╗")
    print(f"  ║  IMU Visualizer running                  ║")
    print(f"  ║  Open: {url:<33s}║")
    print(f"  ║  Press Ctrl+C to stop                    ║")
    print(f"  ╚══════════════════════════════════════════╝\n")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Live 3D IMU/pose visualizer (Foxglove-style)")
    parser.add_argument("npz_file", help="Path to NPZ file")
    parser.add_argument("--port", type=int, default=8888, help="HTTP port (default: 8888)")
    args = parser.parse_args()

    path = Path(args.npz_file)
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)

    print(f"Loading {path} ...")
    serve(str(path), args.port)


if __name__ == "__main__":
    main()
