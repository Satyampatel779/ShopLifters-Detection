const watchlistForm = document.getElementById("watchlist-form");
const videoForm = document.getElementById("video-form");
const watchlistStatus = document.getElementById("watchlist-status");
const watchlistItems = document.getElementById("watchlist-items");
const videoStatus = document.getElementById("video-status");
const videoMeta = document.getElementById("video-meta");
const alertsDiv = document.getElementById("alerts");
const profilesDiv = document.getElementById("profiles");
const livePreview = document.getElementById("live-preview");
const resultVideo = document.getElementById("result-video");
const resultStatus = document.getElementById("result-status");
const statWatchlist = document.getElementById("stat-watchlist");
const statAlerts = document.getElementById("stat-alerts");
const statProfiles = document.getElementById("stat-profiles");
const statStatus = document.getElementById("stat-status");
const liveTelemetry = document.getElementById("live-telemetry");
const activityFeed = document.getElementById("activity-feed");

let currentVideoId = null;
let pollTimer = null;

function logActivity(message) {
  if (!activityFeed) {
    return;
  }
  const ts = new Date().toLocaleTimeString();
  activityFeed.innerHTML = `<div><strong>${ts}</strong> - ${message}</div>${activityFeed.innerHTML}`;
}

function setSystemStatus(text) {
  if (statStatus) {
    statStatus.textContent = text;
  }
}

async function fetchJSON(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

function addAlertCard(alert) {
  const div = document.createElement("div");
  div.className = "alert";
  div.innerHTML = `
    <strong>${alert.watchlist_person_name}</strong> matched in track #${alert.track_id}<br/>
    Score: ${alert.match_score} | Time: ${alert.timestamp_sec}s | ${alert.aisle}
  `;
  alertsDiv.prepend(div);
  logActivity(`Alert triggered for ${alert.watchlist_person_name} in ${alert.aisle}.`);
}

function drawWatchlist(items) {
  watchlistItems.innerHTML = "";
  statWatchlist.textContent = String(items.length);
  if (!items.length) {
    watchlistItems.innerHTML = "<small>No watchlist person added yet.</small>";
    return;
  }
  items.forEach((p) => {
    const badge = document.createElement("span");
    badge.className = "tag";
    badge.textContent = `${p.name}`;
    watchlistItems.appendChild(badge);
  });
}

function drawProfiles(profiles) {
  profilesDiv.innerHTML = "";
  statProfiles.textContent = String(profiles.length);
  if (!profiles.length) {
    profilesDiv.innerHTML = "<small>Profiles will appear after processing starts.</small>";
    return;
  }
  profiles.forEach((p) => {
    const div = document.createElement("div");
    div.className = "profile";
    div.innerHTML = `
      <strong>Track #${p.track_id}</strong><br/>
      Time window: ${p.start_time_sec.toFixed(2)}s - ${p.end_time_sec.toFixed(2)}s<br/>
      Avg speed: ${p.avg_speed_px_per_sec} px/s<br/>
      Aisles visited: ${(p.aisles_visited || []).join(", ") || "N/A"}
    `;
    profilesDiv.appendChild(div);
  });
}

async function refreshWatchlist() {
  const items = await fetchJSON("/api/watchlist");
  drawWatchlist(items);
}

async function refreshAlerts() {
  const alerts = await fetchJSON("/api/alerts");
  alertsDiv.innerHTML = "";
  alerts.forEach(addAlertCard);
  statAlerts.textContent = String(alerts.length);
}

async function loadProfiles(videoId) {
  const profiles = await fetchJSON(`/api/videos/${videoId}/profiles`);
  drawProfiles(profiles);
}

async function pollStatus(videoId) {
  if (pollTimer) {
    clearInterval(pollTimer);
  }
  pollTimer = setInterval(async () => {
    try {
      const job = await fetchJSON(`/api/videos/${videoId}/status`);
      videoStatus.textContent = `Status: ${job.status} | Progress: ${(job.progress * 100).toFixed(1)}%`;
      setSystemStatus(`Pipeline: ${job.status.toUpperCase()}`);

      if (job.summary && Object.keys(job.summary).length > 0) {
        const aisles = (job.summary.aisles || []).map((a) => a.name).join(", ");
        const live = job.summary.live || {};
        const outputVideo = job.summary.output_video || "";
        liveTelemetry.textContent = `Frame ${live.frame_index ?? "-"} | Det ${live.detections ?? 0} | Tracks ${live.active_tracks ?? 0}`;
        videoMeta.innerHTML = `
          <small>
            Profiles: ${job.summary.profiles_count || 0} | Alerts: ${job.summary.alerts_count || 0}<br/>
            Aisles: ${aisles || "N/A"}<br/>
            Live frame: ${live.frame_index ?? "-"} | Detections: ${live.detections ?? 0} | Active tracks: ${live.active_tracks ?? 0}<br/>
            Output file: ${outputVideo || "pending"}
          </small>
        `;
      }
      await loadProfiles(videoId);
      if (job.status === "completed" || job.status === "error") {
        clearInterval(pollTimer);
        await refreshAlerts();
        if (job.status === "completed") {
          resultStatus.textContent = "Loading processed output video...";
          resultVideo.src = `/api/videos/${videoId}/result?t=${Date.now()}`;
          resultVideo.load();
          logActivity("Video processing completed. Rendered output is ready.");
          setSystemStatus("Pipeline: COMPLETED");
        } else {
          setSystemStatus("Pipeline: ERROR");
          logActivity(`Pipeline failed: ${job.error || "unknown error"}.`);
        }
      }
    } catch (err) {
      videoStatus.textContent = err.message;
      clearInterval(pollTimer);
      setSystemStatus("Pipeline: ERROR");
      logActivity(`Status polling failed: ${err.message}.`);
    }
  }, 2000);
}

watchlistForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const formData = new FormData(watchlistForm);
  try {
    const created = await fetchJSON("/api/watchlist", {
      method: "POST",
      body: formData,
    });
    watchlistStatus.textContent = `Added ${created.name} to watchlist.`;
    logActivity(`Watchlist profile added: ${created.name}.`);
    watchlistForm.reset();
    await refreshWatchlist();
  } catch (err) {
    watchlistStatus.textContent = err.message;
    logActivity(`Watchlist upload error: ${err.message}.`);
  }
});

videoForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const formData = new FormData(videoForm);
  try {
    const result = await fetchJSON("/api/videos/upload", {
      method: "POST",
      body: formData,
    });
    currentVideoId = result.video_id;
    videoStatus.textContent = `Video queued. Job ID: ${currentVideoId}`;
    resultStatus.textContent = "Processing in progress. Output video will appear after completion.";
    setSystemStatus("Pipeline: PROCESSING");
    logActivity(`Video job started (${currentVideoId.slice(0, 8)}...).`);
    livePreview.src = `/api/videos/${currentVideoId}/stream?t=${Date.now()}`;
    resultVideo.removeAttribute("src");
    resultVideo.load();
    videoForm.reset();
    await pollStatus(currentVideoId);
  } catch (err) {
    videoStatus.textContent = err.message;
    setSystemStatus("Pipeline: ERROR");
    logActivity(`Video upload error: ${err.message}.`);
  }
});

resultVideo.addEventListener("loadeddata", () => {
  resultStatus.textContent = "Processed output is ready. Click play.";
  logActivity("Processed output video loaded successfully.");
});

resultVideo.addEventListener("error", () => {
  resultStatus.textContent = "Could not load processed output video. Try another upload.";
  logActivity("Processed output failed to load in browser player.");
});

(function connectWebSocket() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/alerts`);

  ws.onopen = () => {
    ws.send("ping");
    setInterval(() => ws.send("ping"), 20000);
  };
  ws.onmessage = (event) => {
    try {
      const alert = JSON.parse(event.data);
      addAlertCard(alert);
    } catch {
      // Ignore non-json heartbeat payloads.
    }
  };
  ws.onclose = () => {
    setTimeout(connectWebSocket, 2000);
  };
})();

(async function init() {
  setSystemStatus("System Ready");
  liveTelemetry.textContent = "Waiting for stream...";
  logActivity("Dashboard initialized.");
  await refreshWatchlist();
  await refreshAlerts();
})();
