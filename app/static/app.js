const refs = [];
const REFERENCE_LIMITS = { image: 9, video: 3, audio: 3 };
const MAX_REFERENCES = 12;
const DRAFT_STORAGE_KEY = "h3-local-ui-composer-draft-v5-local-ready-defaults";
const COMPOSER_SESSION_STORAGE_KEY = "h3-local-ui-composer-session-v1";
const SELECTED_TASK_STORAGE_KEY = "h3-local-ui-selected-task-v1";
const DRAFT_FIELD_IDS = ["prompt", "mode", "duration", "ratio", "preset", "fps", "seed", "guidance", "role", "reference-policy", "stage1-steps", "stage2-steps", "acceleration-mode", "postprocess-1080p"];
// The selected task drives the execution/result panels. The hardware panel
// receives its own active task from api/hardware-status and must never reuse
// this historical selection.
let selectedTask = null;
let activeTask = null;
let selectedTaskSource = "none";
let resultHistory = [];
let resultHistoryLoaded = false;
let selectedResultId = null;
let resultHistoryRequest = null;
const observedResultIds = new Set();
let pendingLatestResult = null;
let pollTimer = null;
let completionHistoryRetryAt = 0;
let composerDefaults = null;
let manualComposerStatePresent = false;
let healthFailureCount = 0;
const displayedProgress = new Map();
const progressEventSeq = new Map();
const displayedProgressEvents = new Map();
const displayedStageProgress = new Map();
const lastStageInfo = new Map();
// Keep a cancellation request independently of the button DOM node: polling
// can briefly return an older running snapshot after the POST is accepted.
const cancelPendingTaskIds = new Set();
let hardwareRequestInFlight = false;
let lastHardwareStatus = null;
let lastHardwareUpdatedAt = null;
let taskEventSource = null;
// The service persists sampling telemetry independently from the human-readable
// status message. Keep the last rendered step only as a same-task polling guard;
// a page refresh always rebuilds it from the persisted task payload.
const displayedSamplingSteps = new Map();
const terminalStates = new Set(["dry_run_complete", "completed", "error", "cancelled"]);
const postprocessTerminalStates = new Set(["completed", "failed", "cancelled"]);
const activeTaskStates = new Set(["queued", "compiling", "ready", "loading", "running", "cancelling"]);
// The product route remains full_quality; the compute kernel is explicit.
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");


function replayVisualState(element, className) {
  if (!element || reducedMotion.matches) return;
  element.classList.remove(className);
  requestAnimationFrame(() => {
    element.classList.add(className);
    element.addEventListener("animationend", () => element.classList.remove(className), { once: true });
  });
}

const stageLabels = {
  queue: "排队等待",
  preparing: "准备任务",
  reference_preprocess: "参考素材预处理",
  qwen: "Qwen 条件理解",
  model_load: "加载主模型",
  sampling: "采样",
  video_vae: "视频 VAE 解码",
  audio_vae: "音频 VAE 解码",
  mp4_export: "MP4 封装",
  stopping: "正在停止",
  completed: "已完成",
  cancelled: "已取消",
  failed: "生成失败"
};
Object.assign(stageLabels, {
  preflight: "高清修复预检",
  restoring: "FlashVSR 超分",
  encoding: "高清版封装",
  stage1_sampling: "第一阶段采样",
  upscale: "3D 潜变量超分",
  stage2: "第二阶段采样",
  video_decode: "视频解码",
  audio_decode: "音频解码",
  mux: "MP4 封装",
  latent_upscale: "3D 潜变量超分",
  stage2_sampling: "第二阶段采样"
});
stageLabels.frame_resize_vae_reencode = "视频放大与重编码";

const $ = (id) => document.getElementById(id);
const statusLabels = { queued: "排队中", compiling: "编译中", loading: "加载 H3", running: "H3 执行中", ready: "计划就绪", dry_run_complete: "参数预览完成", completed: "真实完成", error: "后端错误", cancelled: "已取消" };
const modeNames = { T2V: "文生视频", I2V: "图生视频", R2V: "参考模式", FIRST_LAST_FRAME: "首尾帧过渡" };
const legacyModeIds = { "Text to Video": "T2V", "Image to Video": "I2V", "Reference to Video": "R2V", "First/Last Frame": "FIRST_LAST_FRAME" };
const kernelNames = { kijai_fast: "Kijai 高速内核", official_native: "官方原生内核" };

function comparableKernelId(value) {
  const id = String(value || "").trim().toLowerCase();
  if (!id || id === "auto_shape_vram") return "";
  if (id === "official_native" || id.includes("official_native")) return "official_native";
  if (id === "kijai_fast" || id.startsWith("kijai_")) return "kijai_fast";
  return id;
}

// A cancel request is not a completed cancellation. Keep this separate so
// the page never claims a still-running CUDA kernel has already stopped.
statusLabels.cancelling = "正在停止";

function parseTaskTime(value) {
  const parsed = value ? Date.parse(value) : NaN;
  return Number.isFinite(parsed) ? parsed : null;
}

function formatElapsed(milliseconds) {
  const total = Math.max(0, Math.floor((milliseconds || 0) / 1000));
  const hours = String(Math.floor(total / 3600)).padStart(2, "0");
  const minutes = String(Math.floor((total % 3600) / 60)).padStart(2, "0");
  const seconds = String(total % 60).padStart(2, "0");
  return `${hours}:${minutes}:${seconds}`;
}

function progressReceipt(task) {
  if (task.liveProgressReceipt && typeof task.liveProgressReceipt === "object") return task.liveProgressReceipt;
  const receipt = executionReceipt(task) || {};
  return receipt.progressReceipt && typeof receipt.progressReceipt === "object" ? receipt.progressReceipt : {};
}

function taskTiming(task, now = Date.now()) {
  const timing = task.timing || {};
  const receipt = progressReceipt(task);
  const acceptedAt = parseTaskTime(timing.acceptedAt || task.acceptedAt || task.createdAt);
  const terminal = terminalStates.has(task.state);
  const terminalAt = parseTaskTime(timing.completedAt || task.completedAt || task.updatedAt);
  const end = terminal && terminalAt !== null ? terminalAt : now;
  const liveCurrentStage = task.liveProgressReceipt && task.liveProgressReceipt.currentStage;
  const currentStage = liveCurrentStage || (terminal ? receipt.currentStage : (timing.currentStage || receipt.currentStage));
  const meaningfulStage = receipt.lastMeaningfulStage;
  const stage = terminal
    ? (meaningfulStage || currentStage || timing.currentStage || "preparing")
    : (currentStage || timing.currentStage || meaningfulStage || (task.state === "queued" ? "queue" : "preparing"));
  const stageAt = parseTaskTime(receipt.stageStartedAt || timing.stageStartedAt) || acceptedAt;
  return {
    stage,
    terminal,
    total: acceptedAt === null ? null : Math.max(0, end - acceptedAt),
    stageElapsed: stageAt === null ? null : Math.max(0, end - stageAt),
    acceptedAt,
    end,
    events: Array.isArray(timing.stageEvents) ? timing.stageEvents : [],
    retries: Array.isArray(timing.retryEvents) ? timing.retryEvents : []
  };
}

function isActiveTask(task) {
  return Boolean(task && activeTaskStates.has(task.state));
}

function latestTask(tasks, predicate) {
  return tasks.filter(predicate).sort((a, b) => {
    const aTime = parseTaskTime(a.createdAt) || 0;
    const bTime = parseTaskTime(b.createdAt) || 0;
    return bTime - aTime;
  })[0] || null;
}

function taskOutputPath(task) {
  return task?.result?.outputPath
    || task?.plan?.executionReceipt?.outputPath
    || task?.plan?.compiled?.outputPath
    || "";
}

function isSamplingStage(stage) {
  return stage === "sampling" || stage === "stage1_sampling" || stage === "stage2_sampling";
}

function livePhaseProgress(task) {
  const receipt = task.liveProgressReceipt;
  const step = Number(receipt?.phaseStep);
  const total = Number(receipt?.phaseTotal);
  if (!Number.isFinite(step) || !Number.isFinite(total) || total <= 0) return null;
  const completed = Math.max(0, Math.min(Math.floor(step), Math.floor(total)));
  return { step: completed, total: Math.floor(total), percent: Math.round((completed / total) * 100) };
}

function stageDisplayLabel(stage) {
  const labels = {
    sampling: "一采",
    stage1_sampling: "一采",
    stage2_sampling: "二采",
    frame_resize_vae_reencode: "视频放大重编码",
    video_decode: "视频解码中",
    video_vae: "视频解码中",
    audio_decode: "音频解码中",
    audio_vae: "音频解码中",
    mux: "正在封装MP4",
    mp4_export: "正在封装MP4",
  };
  return labels[stage] || stageLabels[stage] || "处理中";
}

function stageDisplay(task) {
  const stage = taskTiming(task).stage;
  const progress = livePhaseProgress(task);
  const label = stageDisplayLabel(stage);
  const block = denoiserBlockProgress(task, stage);
  const blockLabel = block ? ` · 计算块 ${block.block}/${block.totalBlocks}` : "";
  return progress ? `${label} ${progress.step}/${progress.total} 步${blockLabel} · ${progress.percent}%` : `${label}${blockLabel}`;
}

function samplingProgressEntry(value) {
  const entry = value && value.payload && typeof value.payload === "object" ? value.payload : value;
  if (!entry || typeof entry !== "object") return { seen: false, step: 0, total: 0 };
  const stats = entry.stats && typeof entry.stats === "object" ? entry.stats : {};
  const step = Number(entry.step ?? entry.samplingStep ?? entry.completedSamplingSteps ?? stats.completedSamplingSteps);
  const total = Number(entry.totalSteps ?? entry.requestedSamplingSteps ?? stats.requestedSamplingSteps);
  return { seen: Number.isFinite(step), step: Number.isFinite(step) ? Math.max(0, step) : 0, total: Number.isFinite(total) && total > 0 ? total : 0 };
}

function persistedSamplingProgress(task) {
  const sources = [
    task.samplingTelemetry,
    task.result && task.result.samplingTelemetry,
    task.result && task.result.runtimeStages && task.result.runtimeStages.primary_sampling && task.result.runtimeStages.primary_sampling.samplingTelemetry,
    task.workerRuntimeSnapshot && task.workerRuntimeSnapshot.events
  ];
  let highest = 0;
  let total = 0;
  let seen = false;
  sources.forEach((entries) => {
    if (!Array.isArray(entries)) return;
    entries.forEach((entry) => {
      const normalized = samplingProgressEntry(entry);
      if (!normalized.seen) return;
      seen = true;
      if (normalized.step > highest) highest = normalized.step;
      if (normalized.total > total) total = normalized.total;
    });
  });
  const cancellation = task.cancellation || (task.result && task.result.cancellation) || (task.result && task.result.runtimeStages && task.result.runtimeStages.cancellation) || {};
  const completed = Number(cancellation && cancellation.lastCompletedSamplingStep);
  if (Number.isFinite(completed)) {
    seen = true;
    if (completed > highest) highest = completed;
  }
  return { seen, step: highest, total };
}

function samplingProgress(task, stage) {
  const receipt = progressReceipt(task);
  if (isSamplingStage(stage) && receipt.phaseStep !== undefined && Number(receipt.phaseTotal) > 0) {
    return `${receipt.phaseStep}/${receipt.phaseTotal} 步`;
  }
  if (!isSamplingStage(stage) && !terminalStates.has(task.state)) return "";
  
  // Get total from plan first
  const planTotal = Number(task.plan?.execution?.steps || task.plan?.executionReceipt?.steps);
  
  // Try multiple sources for step number
  let step = 0;
  
  // Source 1: denoiserBlockProgress
  const blockInfo = denoiserBlockProgress(task, "sampling");
  if (blockInfo && blockInfo.step > 0) {
    step = blockInfo.step;
  }
  
  // Source 2: Message parsing
  if (step === 0) {
    const message = String(task.message || "");
    const messageMatch = message.match(/(?:sampling step|第)\s*(\d+)\s*\/\s*(\d+)/i);
    if (messageMatch) {
      step = Number(messageMatch[1]);
    }
  }
  
  // Source 3: Heartbeat
  if (step === 0) {
    const heartbeat = task.runtimeHeartbeat || {};
    const hbPayload = heartbeat.payload || heartbeat;
    step = Number(hbPayload.step || hbPayload.forwardIndex || hbPayload.completedSamplingSteps) || 0;
  }
  
  // Source 4: Latest telemetry
  if (step === 0) {
    const latestTelemetry = Array.isArray(task.samplingTelemetry) && task.samplingTelemetry.length > 0
      ? task.samplingTelemetry[task.samplingTelemetry.length - 1]
      : null;
    if (latestTelemetry) {
      const telemetryPayload = latestTelemetry.payload || latestTelemetry;
      step = Number(telemetryPayload.step || telemetryPayload.forwardIndex || telemetryPayload.completedSamplingSteps) || 0;
    }
  }
  
  // Return formatted string
  if (step > 0 && planTotal > 0) {
    return `${step}/${planTotal} 步`;
  }
  
  return planTotal > 0 ? `--/${planTotal} 步` : "";
}

function denoiserBlockProgress(task, stage) {
  if (!isSamplingStage(stage)) return null;
  const heartbeat = task.runtimeHeartbeat || {};
  const hbPayload = heartbeat.payload || heartbeat;
  const message = String(task.message || "");
  const messageMatch = message.match(/denoiser block\s*(\d+)\s*\/\s*(\d+)/i);
  const block = Number(hbPayload.block ?? messageMatch?.[1]);
  const totalBlocks = Number(hbPayload.totalBlocks ?? messageMatch?.[2]);
  if (!Number.isFinite(block) || !Number.isFinite(totalBlocks) || totalBlocks <= 0) {
    return null;
  }
  return {
    block: Math.max(0, Math.min(block, totalBlocks)),
    totalBlocks,
  };
}

function stageProgressPercent(task, stage, block = null) {
  return livePhaseProgress(task)?.percent ?? null;
}

function executionReceipt(task) {
  const result = task.result || {};
  return result.executionReceipt || task.executionReceipt || task.plan?.executionReceipt || task.plan?.compiled?.executionReceipt || null;
}

function advanceDisplayedProgressEvent(task) {
  const events = Array.isArray(task.progressEvents)
    ? task.progressEvents
        .map((event) => {
          const payload = event?.payload && typeof event.payload === "object" ? event.payload : {};
          const messageMatch = String(event?.message || "").match(/denoiser block\s*(\d+)\s*\/\s*(\d+)\s*returned/i);
          const kind = payload.kind || event?.kind;
          if (kind === "denoiser_block" && payload.event === "returned" && Number.isFinite(Number(payload.block))) {
            return event;
          }
          if (kind === "progress" && messageMatch) {
            return {
              ...event,
              payload: {
                ...payload,
                kind: "denoiser_block",
                event: "returned",
                block: Number(messageMatch[1]),
                totalBlocks: Number(messageMatch[2]),
              },
            };
          }
          return null;
        })
        .filter((event) => event && Number.isFinite(Number(event.seq)))
        .sort((left, right) => Number(left.seq) - Number(right.seq))
    : [];
  if (!progressEventSeq.has(task.id)) progressEventSeq.set(task.id, 0);
  if (!events.length) return;
  const previousSeq = progressEventSeq.get(task.id) || 0;
  const next = events.find((event) => Number(event.seq) > previousSeq);
  if (!next) return;
  progressEventSeq.set(task.id, Number(next.seq));
  displayedProgressEvents.set(task.id, next);
}

function taskDisplaySettings(task) {
  const compiled = task.plan?.compiled || {};
  const receipt = executionReceipt(task) || {};
  const inputScale = receipt.performanceBaseline?.inputScale || {};
  const compiledTiming = compiled.timing || {};
  const compiledCanvas = compiled.canvas || {};
  const requestedResolution = receipt.requestedResolution || compiledCanvas.requestedResolution || {};
  const actualResolution = receipt.effectiveResolution || compiledCanvas.effectiveResolution || requestedResolution;
  const modeId = inputScale.mode || compiled.mode || legacyModeIds[compiled.modeLabel] || "";
  const requestedKernel = receipt.requestedKernel || compiled.requestedKernel || "";
  const actualKernel = receipt.actualKernel || "";
  const requestedFps = Number(receipt.requestedFps ?? inputScale.fps ?? compiledTiming.fps);
  const actualFps = Number(receipt.actualOutputFps ?? receipt.outputFps);
  const duration = Number(inputScale.durationSeconds ?? compiledTiming.exportDurationSeconds ?? compiledTiming.durationSeconds);
  const references = task.plan?.assetSummary?.references || compiled.references || [];
  const requestedSize = requestedResolution.width && requestedResolution.height ? `${requestedResolution.width}×${requestedResolution.height}` : "";
  const actualSize = actualResolution.width && actualResolution.height ? `${actualResolution.width}×${actualResolution.height}` : "";
  const requestedResolutionText = [requestedResolution.preset || compiledCanvas.resolutionPreset, requestedResolution.aspectRatio || compiledCanvas.aspectRatio, requestedSize].filter(Boolean).join(" · ");
  const resolutionMismatch = Boolean(requestedSize && actualSize && requestedSize !== actualSize);
  const comparableRequestedKernel = comparableKernelId(requestedKernel);
  const comparableActualKernel = comparableKernelId(actualKernel);
  const compatibilityFallback = Boolean(
    receipt.compatibilityFallback
    || receipt.kernelBackendReceipt?.compatibilityFallback
    || receipt.fallback === true
  );
  const kernelChanged = Boolean(comparableRequestedKernel && comparableActualKernel && comparableRequestedKernel !== comparableActualKernel);
  const kernelMismatch = kernelChanged && !compatibilityFallback;
  const requestedKernelLabel = kernelNames[comparableRequestedKernel] || kernelNames[requestedKernel] || "未知内核";
  const actualKernelLabel = kernelNames[comparableActualKernel] || kernelNames[actualKernel] || "";
  const displayedKernel = requestedKernel ? requestedKernelLabel : actualKernelLabel;
  const fpsMismatch = Number.isFinite(requestedFps) && Number.isFinite(actualFps) && requestedFps !== actualFps;
  return {
    mode: modeNames[modeId] || "等待确认",
    resolution: resolutionMismatch
      ? `选择 ${requestedResolutionText || "未确认"}；实际执行 ${actualSize || "未确认"}`
      : (requestedResolutionText || actualSize || "等待确认"),
    resolutionMismatch,
    fps: Number.isFinite(requestedFps) ? `${requestedFps} FPS${fpsMismatch ? ` → 实际 ${actualFps} FPS` : ""}` : (Number.isFinite(actualFps) ? `${actualFps} FPS` : "等待确认"),
    fpsMismatch,
    duration: Number.isFinite(duration) ? `${Number.isInteger(duration) ? duration : duration.toFixed(2).replace(/0+$/, "").replace(/\.$/, "")} 秒` : "等待确认",
    kernel: requestedKernel || actualKernel
      ? `${displayedKernel}${kernelChanged ? ` → 实际 ${actualKernelLabel || "未知内核"}${compatibilityFallback ? "（兼容回退）" : ""}` : ""}`
      : "等待确认",
    kernelMismatch,
    assets: references.length ? `${references.length} 个参考素材` : "无参考素材",
  };
}

function twoStageExecutionDetails(receipt = {}) {
  const data = receipt.twoStageSampling || receipt;
  const stage1Steps = data.stage1Steps ?? receipt.stage1Steps;
  const stage2Steps = data.stage2Steps ?? receipt.stage2Steps;
  if (!Number.isFinite(Number(stage1Steps)) || !Number.isFinite(Number(stage2Steps))) return "";
  const combination = `${stage1Steps}+${stage2Steps}`;
  const recoveryErrors = receipt.resourceRecoveryErrors || data.resourceRecoveryErrors || [];
  const rows = [
    ["路线", receipt.routeId || data.routeId || "未记录"],
    ["一采数量", `${stage1Steps} 步`],
    ["二采数量", `${stage2Steps} 步`],
    ["采样组合", combination],
    ["执行方式", data.executionKind || "未记录"],
    ["调用次数", "采样器 " + (data.samplerCalls ?? "未知") + " 次 · 模型生命周期 " + (data.lifecycleCalls ?? "未知") + " 次"],
  ];
  if (recoveryErrors.length) rows.push(["资源恢复记录", recoveryErrors.join("；")]);
  return '<details class="timing-details execution-details"><summary>执行详情</summary><dl>' + rows.map(([label, value]) => '<div><dt>' + escapeHtml(label) + '</dt><dd>' + escapeHtml(value) + '</dd></div>').join("") + '</dl></details>';
}

function executionReceiptMarkup(task) {
  const settings = taskDisplaySettings(task);
  const receipt = executionReceipt(task) || {};
  const items = [["生成模式", settings.mode],["画面规格", settings.resolution, settings.resolutionMismatch],["帧率", settings.fps, settings.fpsMismatch],["视频时长", settings.duration],["计算内核", settings.kernel, settings.kernelMismatch],["参考素材", settings.assets]];
  return '<dl class="task-selection-summary">' + items.map(([label, value, mismatch]) => '<div' + (mismatch ? ' class="is-mismatch"' : "") + '><dt>' + escapeHtml(label) + '</dt><dd>' + escapeHtml(value) + (mismatch ? '<b>不一致</b>' : "") + '</dd></div>').join("") + '</dl>' + twoStageExecutionDetails(receipt);
}

function fitTextToBox(element, { minSize = 10, maxSize = 12 } = {}) {
  if (!element) return false;
  element.style.fontSize = `${maxSize}px`;
  element.style.lineHeight = "1.35";
  let size = maxSize;
  while (element.scrollHeight > element.clientHeight + 1 && size > minSize) {
    size = Math.max(minSize, size - 0.5);
    element.style.fontSize = `${size}px`;
    element.style.lineHeight = `${Math.max(1.08, 1.35 - (maxSize - size) * 0.04)}`;
  }
  const overflowed = element.scrollHeight > element.clientHeight + 1;
  element.dataset.fitState = overflowed ? "overflowed" : "complete";
  return overflowed;
}

const fitTextObservers = new WeakMap();

function observeFitText(element, options) {
  if (!element) return false;
  fitTextToBox(element, options);
  if (typeof ResizeObserver === "undefined" || fitTextObservers.has(element)) return false;
  let frame = 0;
  const observer = new ResizeObserver(() => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => fitTextToBox(element, options));
  });
  observer.observe(element);
  fitTextObservers.set(element, observer);
  return true;
}

function timingMarkup(task) {
  const timing = taskTiming(task);
  const stageLabel = stageDisplay(task);
  const totalLabel = timing.terminal ? "端到端总用时" : "总用时";
  const stageElapsed = timing.stageElapsed === null || timing.stageElapsed <= 0 ? "--:--:--" : formatElapsed(timing.stageElapsed);
  return `<div class="task-timing" data-task-clock="${escapeHtml(task.id)}"><div><small>${totalLabel}</small><strong data-total-clock>${timing.total === null || timing.total <= 0 ? "--:--:--" : formatElapsed(timing.total)}</strong></div><div><small>${timing.terminal ? "完成阶段" : "当前阶段"}</small><strong class="task-stage-value" style="white-space:normal;overflow-wrap:anywhere">${escapeHtml(stageLabel)}</strong></div><div><small>本阶段用时</small><strong data-stage-clock>${stageElapsed}</strong></div></div>`;
}

function timingBreakdownMarkup(task) {
  const timing = taskTiming(task);
  if (!timing.events.length) return "";
  const rows = timing.events.map((event) => {
    const started = parseTaskTime(event.startedAt);
    const ended = parseTaskTime(event.endedAt) || (event.stage === timing.stage ? timing.end : null);
    const elapsed = started !== null && ended !== null ? formatElapsed(ended - started) : "记录中";
    return `<li><span>${escapeHtml(stageLabels[event.stage] || event.stage)}</span><b>${elapsed}</b></li>`;
  }).join("");
  return `<details class="timing-details"><summary>查看真实阶段用时</summary><ul>${rows}</ul><p>排队与重试：${timing.retries.length} 次重试；计时以服务端接受与阶段事件时间戳为准。</p></details>`;
}

function refreshTaskClock() {
  if (!selectedTask) return;
  const timing = taskTiming(selectedTask);
  const card = document.querySelector(`[data-task-clock="${CSS.escape(selectedTask.id)}"]`);
  if (!card) return;
  const total = card.querySelector("[data-total-clock]");
  const stage = card.querySelector("[data-stage-clock]");
  if (total) total.textContent = timing.total === null || timing.total <= 0 ? "--:--:--" : formatElapsed(timing.total);
  if (stage) stage.textContent = timing.stageElapsed === null || timing.stageElapsed <= 0 ? "--:--:--" : formatElapsed(timing.stageElapsed);
}

function installExecutionMode() {
  const select = document.createElement("select");
  select.id = "execution-mode";
  select.innerHTML = `<option value="real">直接生成</option><option value="dry-run">参数预览</option>`;
  select.className = "execution-mode";
  const actionRow = $("generate").parentElement;
  actionRow.insertBefore(select, $("generate"));
  const generateLabel = $("generate").querySelector(".generate-label");
  if (generateLabel) generateLabel.textContent = "开始生成";
  else $("generate").textContent = "开始生成";
}

async function refreshBackendStatus() {
  try {
    const response = await fetch("api/backend/health");
    const data = await response.json();
    document.querySelector(".status-pill").textContent = data.online ? `● 独立 H3 后端 ${data.torch || "online"}` : "● 后端 offline";
  } catch (_error) {
    document.querySelector(".status-pill").textContent = "● H3 后端未初始化";
  }
}

function installReferencePolicyControl() {
  const fields = document.querySelector(".advanced-fields");
  if (!fields || $("reference-policy")) return;
  const wrapper = document.createElement("div");
  wrapper.innerHTML = `<label class="field-label" for="reference-policy">参考范围</label><select id="reference-policy"><option value="complete" selected>完整（推荐）</option><option value="balanced_5090">均衡建议（会缩短区间）</option><option value="fast_5090">快速参考（会缩短区间）</option></select><small>默认保留你选定的视频区间；只有主动选择快速/均衡才会缩短参考范围。</small>`;
  fields.appendChild(wrapper);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[char]));
}

const modeErrorMessages = {
  "T2V does not accept reference materials": "文生视频模式不接受参考素材，请移除参考素材或切换生成模式。",
  "I2V requires at least one picture reference": "图生视频模式至少需要一个图片参考素材。",
  "First / Last Frame requires exactly two picture references": "首尾帧模式必须正好提供两个图片参考素材。",
  "First / Last Frame accepts only the two picture references": "首尾帧模式只接受两个图片参考素材，请移除其他素材。",
  "R2V requires at least one reference material": "参考模式至少需要一个参考素材。"
};

const modeReferenceGuidance = {
  R2V: "参考模式需要至少一项图片或视频素材；仅音频会按官方规则阻断。",
  T2V: "文生视频不需要参考素材，也不接受参考素材。",
  I2V: "图生视频需要至少一张图片素材。",
  FIRST_LAST_FRAME: "首尾帧需要正好两张图片素材。",
};
const zeroReferenceRedirectMessage = "未检测到参考素材，已按官方规则切换为文生视频。";
const referenceModeRedirectMessage = "检测到参考素材，已自动切换为参考模式。";
let modeSyncInProgress = false;

// Keep the visible selector, the native value and the eventual request in
// lockstep. Only T2V/R2V participate; I2V and 首尾帧 remain user-controlled.
function syncGenerationModeFromReferences() {
  const mode = $("mode");
  if (!mode || modeSyncInProgress) return;
  const shouldBe = refs.length > 0 ? "R2V" : "T2V";
  if (mode.value !== "T2V" && mode.value !== "R2V") return;
  if (mode.value === shouldBe) return;
  modeSyncInProgress = true;
  try {
    mode.value = shouldBe;
    mode.dispatchEvent(new Event("change", { bubbles: true }));
    const visibleMode = mode.nextElementSibling?.querySelector?.(".hover-select-trigger");
    const expectedLabel = mode.options[mode.selectedIndex]?.textContent || "";
    if (visibleMode && visibleMode.textContent !== expectedLabel) visibleMode.textContent = expectedLabel;
    setModeFeedback(shouldBe === "R2V" ? referenceModeRedirectMessage : zeroReferenceRedirectMessage);
  } finally {
    modeSyncInProgress = false;
  }
}

const backendErrorMessages = [
  [/^full_quality requires steps/i, "高质量路线步数无效，请选择 20、30、40、50 或 60 步。"],
  [/^steps must be one of /i, "采样步数无效，请选择 20、30、40、50 或 60 步。"],
  [/^stage1Steps must be one of /i, "一采步数无效，请选择 20、30、40、50 或 60 步。"],
  [/^stage2Steps must be one of /i, "二采步数无效，请选择 3、5、7 或 10 步。"],
  [/^memoryStrategy must be /i, "显存策略无效，请选择自动。"],
  [/^ffnChunk(?:s)? must be /i, "计算分块设置无效，请重新选择计算分块。"],
  [/^ffnChunk and ffnChunks conflict$/i, "计算分块设置冲突，请只保留一个分块设置。"],
  [/^assetPrecision must be /i, "素材精度设置无效，请重新选择素材精度。"],
  [/^fps must be exactly 24$/i, "帧率必须是 24 FPS。"],
  [/^durationSeconds must be between ([\d.]+) and ([\d.]+)$/i, (_, min, max) => `时长必须在 ${min} 到 ${max} 秒之间。`],
  [/^executionMode must be /i, "执行方式无效，请重新提交。"],
  [/^non-24 FPS routes are retired/i, "非 24 FPS 路线已停用，请使用 24 FPS。"],
  [/^route_removed\/unsupported_route/i, "当前生成路线已停用，请重新选择生成路线。"],
  [/^task not found$/i, "找不到对应任务，可能已经被清理。"],
  [/^postprocess task not found$/i, "找不到对应的高清修复任务。"],
  [/^endpoint not found$/i, "当前服务不支持这个操作。"],
  [/^request body must contain valid JSON$/i, "请求格式无效，请重新提交。"],
  [/^upload must use multipart\/form-data$/i, "上传格式无效，请重新选择文件。"],
  [/^server error:/i, "服务器暂时出错，请稍后重试。"],
];

function userErrorMessage(message = "") {
  const text = String(message || "");
  if (modeErrorMessages[text]) return modeErrorMessages[text];
  if (text === "mode must be T2V, I2V, FIRST_LAST_FRAME, or R2V") return "生成模式无效，请重新选择生成模式。";
  for (const [pattern, replacement] of backendErrorMessages) {
    if (pattern.test(text)) return typeof replacement === "function" ? replacement(...text.match(pattern).slice(1)) : replacement;
  }
  if (/[A-Za-z]/.test(text) && !/[一-龥]/.test(text)) return "操作失败，请检查设置后重试。";
  return text;
}

function setError(message = "") {
  const displayMessage = userErrorMessage(message);
  $("error").hidden = !displayMessage;
  $("error").textContent = displayMessage;
}

function setModeFeedback(message = "") {
  const feedback = $("mode-feedback");
  if (!feedback) return;
  feedback.hidden = !message;
  feedback.textContent = message;
}

function renderReferenceCapacity(counts) {
  const actualCounts = counts || refs.reduce((result, ref) => {
    result[ref.kind] = (result[ref.kind] || 0) + 1;
    return result;
  }, { image: 0, video: 0, audio: 0 });
  const guidance = modeReferenceGuidance[$("mode").value] || "";
  $("reference-capacity").textContent = `图片 ${actualCounts.image}/${REFERENCE_LIMITS.image} · 视频 ${actualCounts.video}/${REFERENCE_LIMITS.video} · 音频 ${actualCounts.audio}/${REFERENCE_LIMITS.audio}；视频内含原声不另占上传名额。${guidance ? ` ${guidance}` : ""}`;
}

function detectKind(file) {
  if (file.type.startsWith("image/")) return "image";
  if (file.type.startsWith("video/")) return "video";
  if (file.type.startsWith("audio/")) return "audio";
  return null;
}

function readVideoMetadata(file) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.preload = "metadata";
    video.onloadedmetadata = () => {
      const duration = Number.isFinite(video.duration) ? video.duration : null;
      URL.revokeObjectURL(url);
      resolve(duration ? { originalDurationSeconds: duration, sourceFrameCount: Math.floor(duration * 24) } : {});
    };
    video.onerror = () => { URL.revokeObjectURL(url); resolve({}); };
    video.src = url;
  });
}

function formatBytes(size) {
  if (!size) return "0 B";
  if (size < 1024 * 1024) return `${Math.round(size / 1024)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

function releaseReferencePreview(ref) {
  if (ref && ref.previewUrl) URL.revokeObjectURL(ref.previewUrl);
  if (ref) ref.previewUrl = null;
}

function referenceToken(ref, index) {
  const ordinal = refs.slice(0, index + 1).filter((item) => item.kind === ref.kind).length;
  const labels = { image: "Picture", video: "Video", audio: "Audio" };
  return `${labels[ref.kind] || "Reference"} ${ordinal}`;
}

function saveComposerDraft() {
  try {
    const fields = {};
    document.querySelectorAll("input[id], select[id], textarea[id]").forEach((field) => {
      if (field.id === "file-input") return;
      if (field.type === "radio" || field.type === "checkbox") fields[field.id] = Boolean(field.checked);
      else fields[field.id] = field.value;
    });
    const persistedReferences = refs.map(({ previewUrl, ...reference }) => reference);
    localStorage.setItem(DRAFT_STORAGE_KEY, JSON.stringify({ version: 5, fields, references: persistedReferences }));
    manualComposerStatePresent = true;
  } catch (_error) {
    // A private browser session can deny storage. The live composer remains usable.
  }
}

function refreshComposerControls() {
  document.querySelectorAll("select[id]").forEach((select) => {
    select.dispatchEvent(new Event("change", { bubbles: false }));
  });
}

function captureComposerDefaults() {
  composerDefaults = Object.fromEntries(DRAFT_FIELD_IDS.map((id) => {
    const field = $(id);
    return [id, field ? (field.type === "checkbox" ? Boolean(field.checked) : field.value) : undefined];
  }).filter(([, value]) => value !== undefined));
}

function resetComposerToDefaults() {
  refs.splice(0).forEach(releaseReferencePreview);
  Object.entries(composerDefaults || {}).forEach(([id, value]) => {
    const field = $(id);
    if (field) {
      if (field.type === "checkbox") field.checked = Boolean(value);
      else field.value = value;
    }
  });
  $("file-input").value = "";
  localStorage.removeItem(DRAFT_STORAGE_KEY);
  renderReferences();
}

function restoreComposerForCurrentSession() {
  try {
    if (sessionStorage.getItem(COMPOSER_SESSION_STORAGE_KEY) !== "active") {
      sessionStorage.setItem(COMPOSER_SESSION_STORAGE_KEY, "active");
    }
    // 人工最后一次设置是刷新后的唯一恢复来源；不要用默认值或历史任务覆盖它。
    restoreComposerDraft();
  } catch (_error) {
    // Storage can be unavailable in a private browser session; keep page defaults.
  }
}

function setComposerFieldFromTask(id, value) {
  if (value === undefined || value === null || value === "") return;
  const field = $(id);
  if (!field || !field.options) return;
  const stringValue = String(value);
  if ([...field.options].some((option) => option.value === stringValue)) {
    field.value = stringValue;
    const picker = field.nextElementSibling;
    const trigger = picker && picker.querySelector ? picker.querySelector(".hover-select-trigger") : null;
    const selected = field.options[field.selectedIndex];
    if (trigger) trigger.textContent = selected ? selected.textContent : "请选择";
    if (picker && picker.querySelectorAll) picker.querySelectorAll(".hover-select-option").forEach((option) => {
      const active = option.dataset.value === field.value;
      option.classList.toggle("is-selected", active);
      option.setAttribute("aria-selected", String(active));
    });
  }
}

function hydrateComposerFromTask(task) {
  const compiled = task?.plan?.compiled || {};
  const receipt = executionReceipt(task) || {};
  const inputScale = receipt.performanceBaseline?.inputScale || {};
  const timing = compiled.timing || {};
  const canvas = compiled.canvas || {};
  const requested = receipt.requestedResolution || canvas.requestedResolution || {};
  const mode = compiled.mode || legacyModeIds[compiled.modeLabel];
  const duration = inputScale.durationSeconds ?? timing.durationSeconds;
  const stage1Steps = receipt.stage1Steps ?? compiled.advanced?.stage1Steps ?? 20;
  const stage2Steps = receipt.stage2Steps ?? compiled.advanced?.stage2Steps ?? 3;
  setComposerFieldFromTask("stage1-steps", stage1Steps);
  setComposerFieldFromTask("stage2-steps", stage2Steps);
  setComposerFieldFromTask("mode", mode);
  setComposerFieldFromTask("ratio", requested.aspectRatio || canvas.aspectRatio);
  setComposerFieldFromTask("preset", requested.preset || canvas.resolutionPreset);
  setComposerFieldFromTask("duration", duration);
  renderReferences();
}

async function fetchTaskDetail(task) {
  if (!task || !task.id) return task;
  try {
    const response = await fetch("api/tasks/" + encodeURIComponent(task.id));
    return response.ok ? await response.json() : task;
  } catch (_error) {
    return task;
  }
}

function observeServiceLifecycle(available) {
  if (available) {
    healthFailureCount = 0;
    return;
  }
  healthFailureCount += 1;
}

function restoreComposerDraft() {
  try {
    const rawDraft = localStorage.getItem(DRAFT_STORAGE_KEY) || localStorage.getItem("h3-local-ui-composer-draft-v1");
    const draft = JSON.parse(rawDraft || "null");
    if (!draft || ![1, 2, 3, 4, 5].includes(draft.version)) return;
    manualComposerStatePresent = true;
    let migratedKernel = false;
    Object.entries(draft.fields || {}).forEach(([id, value]) => {
      const field = $(id);
      if (!field) return;
      if (field.type === "radio" || field.type === "checkbox") {
        field.checked = Boolean(value);
      } else if (typeof value === "string") {
        if (field.options) {
          if ([...field.options].some((option) => option.value === value)) setComposerFieldFromTask(id, value);
          else if (id === "acceleration-mode") {
            setComposerFieldFromTask(id, "kijai_fast");
            migratedKernel = true;
          }
        } else field.value = value;
      }
    });
    refreshComposerControls();
    if (migratedKernel) saveComposerDraft();
    const restoredReferences = Array.isArray(draft.references) ? draft.references.slice(0, MAX_REFERENCES) : [];
    restoredReferences.forEach((reference) => {
      if (reference && REFERENCE_LIMITS[reference.kind] && refs.filter((item) => item.kind === reference.kind).length < REFERENCE_LIMITS[reference.kind]) {
        refs.push({ ...reference, previewUrl: null });
      }
    });
  } catch (_error) {
    // Ignore a stale or malformed local draft instead of blocking the workbench.
  }
}

function referencePlanSummary(compiled) {
  const planning = compiled.referencePlanning || {};
  const plans = planning.videos || [];
  if (!plans.length) return "";
  const policy = planning.policyPreset || "complete";
  const label = policy === "complete" ? "完整参考范围" : `${policy}（主动选择的性能策略）`;
  const override = planning.policyOverridden ? "；已使用高级自定义预算" : "";
  return `<div class="reference-plan"><strong>参考视频计划 · ${escapeHtml(label)}</strong>${plans.map((plan) => `<div><span>${escapeHtml(plan.token)} ${escapeHtml(plan.name)}</span><span>${plan.selectedInterval.startSeconds}s–${plan.selectedInterval.endSeconds}s · 实际 ${plan.h3FrameCount} 合法帧 · ${plan.officialResize.width}×${plan.officialResize.height} 保持比例不裁剪 · 2fps语义 ${plan.semanticSampling.localFrameIndices.length} 帧${plan.truncated ? ` · ${escapeHtml(plan.truncationReasons.join("；"))}` : ""}</span></div>`).join("")}<small>实际使用 ${planning.totalBudgetFrames} 帧${override}；合法网格未采用的末尾只提示，不阻断生成。</small></div>`;
}

function renderReferences() {
  $("ref-count").textContent = `${refs.length} / ${MAX_REFERENCES}`;
  renderReferenceCapacity();
  const referenceVideoFps = $("reference-video-fps");
  if (referenceVideoFps) {
    const hasVideo = refs.some((ref) => ref.kind === "video");
    // Keep this as a normal production parameter: users may preselect it.
    // Without a video reference the value is simply omitted from the payload.
    referenceVideoFps.disabled = false;
  }
  const referenceList = $("reference-list");
  const assetCount = Math.max(refs.length, 1);
  referenceList.style.setProperty("--asset-count", String(assetCount));
  referenceList.style.setProperty("--asset-gap-total", `${Math.max(0, assetCount - 1) * 4}px`);
  referenceList.innerHTML = refs.map((ref, index) => {
    const token = referenceToken(ref, index);
    const preview = ref.kind === "image" && ref.previewUrl
      ? `<img class="ref-image-preview" src="${escapeHtml(ref.previewUrl)}" alt="${escapeHtml(token)} 本地预览">`
      : `<div class="ref-media-fallback ${escapeHtml(ref.kind)}" aria-hidden="true">${ref.kind === "video" ? "VIDEO" : ref.kind === "audio" ? "AUDIO" : "IMAGE"}</div>`;
    const segmentFields = ref.kind === "video" ? `<div class="segment-fields"><input class="video-segment" data-index="${index}" data-field="startSeconds" type="number" min="0" step="0.01" placeholder="起点 s" value="${ref.startSeconds ?? ""}"><input class="video-segment" data-index="${index}" data-field="endSeconds" type="number" min="0" step="0.01" placeholder="终点 s" value="${ref.endSeconds ?? ""}"></div>` : "";
    return `<article class="ref-card ref-card-${escapeHtml(ref.kind)}"><div class="ref-card-media">${preview}<span class="ref-order">${escapeHtml(token)}</span></div><div class="ref-card-body"><strong>${escapeHtml(ref.name)}</strong><small>${escapeHtml(ref.kind.toUpperCase())} · ${formatBytes(ref.size)}${ref.originalDurationSeconds ? ` · ${Number(ref.originalDurationSeconds).toFixed(2)}s` : ""}</small>${segmentFields}</div><button class="remove-ref" data-index="${index}" type="button" aria-label="移除 ${escapeHtml(token)}：${escapeHtml(ref.name)}">×</button></article>`;
  }).join("");
  const tray = $("reference-tray");
  if (tray) tray.hidden = false;
  document.querySelectorAll(".remove-ref").forEach((button) => button.addEventListener("click", (event) => {
    event.stopPropagation();
    const index = Number(button.dataset.index);
    const removed = refs[index];
    if (removed) releaseReferencePreview(removed);
    refs.splice(index, 1);
    renderReferences();
    saveComposerDraft();
  }));
  document.querySelectorAll(".video-segment").forEach((input) => input.addEventListener("change", () => {
    const ref = refs[Number(input.dataset.index)];
    const value = input.value.trim();
    if (ref) ref[input.dataset.field] = value === "" ? null : Number(value);
  }));
  syncGenerationModeFromReferences();
}

function selectedCanvasDimensions() {
  const ratio = $("ratio").value;
  const preset = $("preset").value;
  const fixed = {
    "1080p": {
      "16:9": [1920, 1088], "9:16": [1088, 1920], "1:1": [1088, 1088],
      "4:3": [1456, 1088], "3:4": [1088, 1456],
    },
    "720p": {
      "16:9": [1248, 704], "9:16": [704, 1248], "1:1": [704, 704],
      "4:3": [928, 704], "3:4": [704, 928],
    },
    "768p": {
      "16:9": [1344, 768], "9:16": [768, 1344], "1:1": [768, 768],
      "4:3": [1024, 768], "3:4": [768, 1024],
    },
    "640p": {
      "16:9": [1152, 640], "9:16": [640, 1152], "1:1": [640, 640],
      "4:3": [864, 640], "3:4": [640, 864],
    },
    "576p": {
      "16:9": [1024, 576], "9:16": [576, 1024], "1:1": [576, 576],
      "4:3": [768, 576], "3:4": [576, 768],
    },
    "480p": {
      "16:9": [848, 480], "9:16": [480, 848], "1:1": [480, 480],
      "4:3": [640, 480], "3:4": [480, 640],
    },
  };
  return fixed[preset]?.[ratio] || null;
}

function exactSeedValue() {
  const rawSeed = $("seed").value.trim();
  if (!rawSeed) return null;
  if (!/^[0-9]+$/.test(rawSeed)) throw new Error("随机种子必须是 0 到 9223372036854775807 之间的整数。");
  const seed = BigInt(rawSeed);
  if (seed > 9223372036854775807n) throw new Error("随机种子必须是 0 到 9223372036854775807 之间的整数。");
  return seed.toString();
}

function payload() {
  const seed = exactSeedValue();
  const stage1Steps = Number($("stage1-steps").value);
  const stage2Steps = Number($("stage2-steps").value);
  const dimensions = selectedCanvasDimensions();
  return {
    prompt: $("prompt").value,
    mode: $("mode").value,
    durationSeconds: Number($("duration").value),
    fps: Number($("fps").value),
    aspectRatio: $("ratio").value,
    resolutionPreset: $("preset").value,
    ...(dimensions ? { exportWidth: dimensions[0], exportHeight: dimensions[1] } : {}),
    references: refs.map((ref) => ({ name: ref.name, kind: ref.kind, size: ref.size, mimeType: ref.mimeType, clientMimeType: ref.clientMimeType, role: ref.role, path: ref.path || "", assetId: ref.assetId || "", sha256: ref.sha256 || "", mediaFacts: ref.mediaFacts || {}, startSeconds: ref.startSeconds, endSeconds: ref.endSeconds })),
    // Keep seed as a decimal string until the Python compiler parses it.
    // JavaScript Number cannot represent the full signed 63-bit seed range.
    advanced: { seed, stage1Steps, stage2Steps, assetPrecision: "fixed_optimized", modelQuant: "int8", referenceRole: $("role").value, referencePolicyPreset: $("reference-policy") ? $("reference-policy").value : "complete", accelerationMode: "full_quality", requestedKernel: $("acceleration-mode") ? $("acceleration-mode").value : "kijai_fast", ...(refs.some((ref) => ref.kind === "video") ? { reference_video_fps: Number($("reference-video-fps")?.value || 2), reference_video_vae_fps: Number($("reference-video-vae-fps")?.value || 12) } : {}) },
    executionMode: $("execution-mode") ? $("execution-mode").value : "real"
    , postprocess1080p: Boolean($("postprocess-1080p")?.checked)
  };
}

const nvidiaVsrTasks = new Map();
const nvidiaVsrPollers = new Map();
function nvidiaVsrPreviewUrl(task) {
  const output = String(task?.outputPath || "").replace(/^output\//, "");
  return output ? `api/output/${output.split("/").map(encodeURIComponent).join("/")}` : "";
}
function renderNvidiaVsrStatus(sourceTask) {
  const result = $("result");
  if (!result) return;
  const task = nvidiaVsrTasks.get(sourceTask?.id);
  const card = document.querySelector(`.queue-card[data-task-id="${CSS.escape(sourceTask?.id || "")}"]`);
  const status = card?.querySelector("[data-vsr-status]");
  if (status) {
    status.hidden = !sourceTask?.postprocess1080p || !task;
    status.textContent = task ? `1080P超分·${Math.max(0, Math.min(100, Number(task.progress) || 0))}%` : "";
  }
  if (!sourceTask?.postprocess1080p || !task) return;
  if (task.state === "completed" && task.outputAuthentic === true && nvidiaVsrPreviewUrl(task)) {
    const video = result.querySelector("video.result-video");
    if (video) {
      const previewUrl = nvidiaVsrPreviewUrl(task);
      if (video.src !== new URL(previewUrl, window.location.href).href) {
        video.src = previewUrl;
        video.dataset.previewKey = `${sourceTask.id}:nvidia-vsr:${task.id}`;
        video.load();
      }
      video.style.aspectRatio = "1920 / 1088";
      const note = result.querySelector(".result-preview-note");
      if (note) note.textContent = "NVIDIA VSR输出 · 1920×1088";
    }
  }
}
async function startNvidiaVsr(sourceTask) {
  if (!sourceTask?.postprocess1080p || sourceTask.state !== "completed" || sourceTask.result?.outputAuthentic !== true || nvidiaVsrTasks.has(sourceTask.id)) return;
  const response = await fetch("api/nvidia-vsr/tasks", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({sourceTaskId:sourceTask.id})});
  const task = await response.json(); if (!response.ok) throw new Error(task.error || "NVIDIA VSR任务创建失败");
  nvidiaVsrTasks.set(sourceTask.id, task); renderNvidiaVsrStatus(sourceTask);
  const poll = setInterval(async () => { const r = await fetch(`api/nvidia-vsr/tasks/${encodeURIComponent(task.id)}`); if (!r.ok) return; const next = await r.json(); nvidiaVsrTasks.set(sourceTask.id, next); renderNvidiaVsrStatus(sourceTask); if (["completed","failed","cancelled"].includes(next.state)) { clearInterval(poll); nvidiaVsrPollers.delete(sourceTask.id); } }, 1000);
  nvidiaVsrPollers.set(sourceTask.id, poll);
}

function compiledModeTitle(compiled = {}) {
  return [compiled.mode, compiled.modeLabel].filter(Boolean).join(" · ") || "生成任务";
}

function renderCompiled(compiled, taskId = "preview") {
  const plan = { taskId, dryRun: true, compiled };
  const order = (compiled.conditioningOrder || []).map((item) => { const ref = (compiled.references || []).find((value) => value.inputIndex === item.inputIndex) || {}; return `<li><strong>${escapeHtml(item.token)} ${escapeHtml(item.name)}</strong> · ${escapeHtml(item.kind)}${ref.preprocessPlan ? ` · ${escapeHtml(ref.preprocessPlan.mode)}` : ""}${item.pairedEmbeddedAudio ? " · 使用同区间内嵌音轨" : ""}</li>`; }).join("");
  const route = compiled.algorithmRoute?.algorithmRoute || "--";
  const routeText = route === "low_fps_time_remap_experimental"
    ? `实验减负路线 · 时间缩放 ${compiled.timing.timeScale}× · 非官方原生 FPS`
    : "官方质量路线 · 24 FPS 原样";
  $("result").innerHTML = `<div class="queue-meta"><strong>${escapeHtml(compiledModeTitle(compiled))} · ${compiled.routing.primaryModel}</strong><span class="state">参数已验证</span></div><p class="hint">${escapeHtml(routeText)}<br>${escapeHtml(compiled.timing.formula)} = ${compiled.timing.frameCount} 模型帧 · 视频 latent ${compiled.timing.videoLatentT} · 音频 latent ${compiled.timing.audioLatentT} · 导出 ${compiled.timing.exportFrameCount} 帧 / ${compiled.timing.exportDurationSeconds}s · ${compiled.canvas.internal.width}×${compiled.canvas.internal.height} → ${compiled.canvas.export.width}×${compiled.canvas.export.height}</p>${order ? `<section class="conditioning-order"><strong>模型实际采用顺序</strong><ol>${order}</ol></section>` : ""}${referencePlanSummary(compiled)}<details class="diagnostic-details"><summary>查看参数明细</summary><pre class="plan-view">${escapeHtml(JSON.stringify(plan, null, 2))}</pre></details>`;
}

async function compile() {
  setError("");
  const requestPayload = payload();
  const response = await fetch("api/compile", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(requestPayload) });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "参数编译失败");
  renderCompiled(data.compiled);
  return data.compiled;
}

// Live tasks show status and diagnostics separately; the raw manifest stays collapsed.
function renderQueueLegacy(task) {
  selectedTask = task;
  const queueState = $("queue-state");
  if (queueState) queueState.textContent = statusLabels[task.state] || task.state;
  const canCancel = ["queued", "compiling", "ready", "loading", "running"].includes(task.state);
  $("queue").innerHTML = `<div class="queue-card"><div class="queue-meta"><strong>${escapeHtml(task.id)} · ${escapeHtml(compiledModeTitle(task.plan.compiled))}</strong><span class="state">${escapeHtml(statusLabels[task.state] || task.state)}</span></div><div class="progress"><i style="width:${task.progress}%"></i></div><div class="queue-foot"><span>${escapeHtml(task.message)} · ${task.progress}%</span>${canCancel ? `<button class="cancel-button" id="cancel-task">取消</button>` : ""}</div>${timingMarkup(task)}</div>`;
  const receivedProgress = Math.max(0, Math.min(100, Number(task.progress) || 0));
  const displayProgress = Math.max(receivedProgress, displayedProgress.get(task.id) || 0);
  displayedProgress.set(task.id, displayProgress);
  const progressBar = $("queue").querySelector(".progress i");
  const progressLabel = $("queue").querySelector(".queue-foot span");
  if (progressBar) progressBar.style.width = `${displayProgress}%`;
  if (progressLabel) progressLabel.textContent = `${task.message || ""} · ${displayProgress}%`;
  const cancel = $("cancel-task");
  if (cancel) {
    cancel.textContent = "停止生成";
    cancel.setAttribute("aria-label", "停止生成");
    cancel.addEventListener("click", cancelTask);
  }
}

function createQueueCard(task) {
  const queue = $("queue");
  queue.replaceChildren();
  const card = document.createElement("div");
  card.className = "queue-card";
  card.dataset.taskId = task.id;
  card.innerHTML = `<div class="queue-meta"><strong data-queue-title></strong><div class="queue-meta-actions"><span class="state" data-queue-state></span><button class="cancel-button" id="cancel-task" type="button">停止生成</button></div></div><div class="queue-progress"><div class="progress-head"><span>总进度</span><strong data-total-progress>0%</strong></div><div class="progress"><i></i></div></div><div class="queue-live-output"><div class="queue-live-copy"><div class="queue-foot"><span data-queue-message></span></div><div class="vsr-status" data-vsr-status hidden></div><div data-queue-receipt></div></div></div><div data-queue-timing></div>`;
  card.querySelector("#cancel-task").addEventListener("click", cancelTask);
  queue.append(card);
  return card;
}

function taskProgressMessage(task, { cancelPending = false } = {}) {
  if (cancelPending && !terminalStates.has(task.state)) return "正在停止，等待当前步骤结束";
  if (task.state === "completed") {
    const stage = progressReceipt(task).lastMeaningfulStage;
    return stage ? `视频已生成 · 最后阶段：${stageLabels[stage] || stage}` : "视频已生成";
  }
  if (task.state === "dry_run_complete") return "参数预览完成";
  if (task.state === "cancelled") return "任务已停止";
  if (task.state === "error") return "生成失败，请查看结果区";
  return stageDisplay(task);
}

// A task card remains mounted throughout its lifetime. Polling only mutates
// children, so a normal click cannot be lost between mousedown and mouseup.
function renderQueue(task) {
  const queue = $("queue");
  let card = queue.querySelector(".queue-card");
  if (!card || card.dataset.taskId !== task.id) card = createQueueCard(task);
  const isTerminal = terminalStates.has(task.state);
  const stateChanged = card.dataset.taskState !== task.state;
  card.dataset.taskState = task.state;
  document.body.dataset.taskActive = String(activeTaskStates.has(task.state));
  if (isTerminal) cancelPendingTaskIds.delete(task.id);
  const cancelPending = cancelPendingTaskIds.has(task.id);
  const canCancel = ["queued", "compiling", "ready", "loading", "running"].includes(task.state);
  const historical = selectedTaskSource === "history" && activeTask && activeTask.id !== task.id;
  const stateLabel = historical
    ? `历史任务 · ${statusLabels[task.state] || task.state}`
    : (cancelPending && !isTerminal ? "正在停止" : (statusLabels[task.state] || task.state));
  const message = taskProgressMessage(task, { cancelPending });
  const displaySettings = taskDisplaySettings(task);
  const queueState = $("queue-state");
  if (queueState) queueState.textContent = stateLabel;
  card.querySelector("[data-queue-title]").textContent = `任务 ${task.id} · ${displaySettings.mode}`;
  card.querySelector("[data-queue-state]").textContent = stateLabel;
  const receivedProgress = Math.max(0, Math.min(100, Number(task.progress) || 0));
  card.querySelector(".progress i").style.width = `${receivedProgress}%`;
  card.querySelector("[data-total-progress]").textContent = `${receivedProgress}%`;
  card.querySelector("[data-queue-message]").textContent = message;
  renderNvidiaVsrStatus(task);
  card.querySelector("[data-queue-receipt]").innerHTML = `${executionReceiptMarkup(task)}${assetReceiptMarkup(task)}`;
  card.querySelector("[data-queue-timing]").innerHTML = timingMarkup(task);
  const liveCopy = card.querySelector(".queue-live-copy");
  observeFitText(liveCopy, { minSize: 10, maxSize: 12 });
  observeFitText(card.querySelector(".task-stage-value"), { minSize: 10, maxSize: 11 });
  const cancel = card.querySelector("#cancel-task");
  const showCancel = canCancel || cancelPending || task.state === "cancelling";
  cancel.hidden = !showCancel;
  cancel.disabled = cancelPending || task.state === "cancelling";
  cancel.textContent = cancelPending || task.state === "cancelling" ? "正在停止" : "停止生成";
  cancel.setAttribute("aria-label", cancel.textContent);
  if (stateChanged) replayVisualState(card, "state-transition");
}

function assetReceiptMarkup(task) {
  return "";
}

function repairTargetMarkup({ disabled = false, activeTarget = "" } = {}) {
  const unavailable = !FLASHVSR_AVAILABLE;
  const buttonLabel = unavailable ? "新超分模型准备中" : "修复当前视频 · 1080p";
  return `<div class="repair-target-group" aria-label="修复当前生成视频">
    <button class="repair-target-button ${activeTarget === "1080p" ? "is-active" : ""}" type="button" data-repair-target="1080p" ${(disabled || unavailable) ? "disabled" : ""}>${buttonLabel}</button>
  </div>`;
}

function resultSwitcherMarkup(results = [], selectedId = "", { disabled = false } = {}) {
  const options = results.length
    ? results.map((item) => `<option value="${escapeHtml(item.id)}" ${selectedId === item.id ? "selected" : ""}>${escapeHtml(resultHistoryLabel(item))}</option>`).join("")
    : `<option value="">暂无真实产物</option>`;
  return `<div class="result-switcher"><label><span>真实产物</span><select data-result-switcher ${disabled || !results.length ? "disabled" : ""}>${options}</select></label></div>`;
}

function usageGuideMarkup() {
  return `<div class="usage-guide-heading"><strong>功能说明</strong><span>每项设置都会影响速度、质量或显存</span></div><div class="usage-guide-list"><div><b>一采说明</b><span>步数越高，细节和稳定性通常更好，但速度更慢；想快速试效果可降低步数。</span></div><div><b>时长说明</b><span>经测试，任何分辨率下超过 10 秒都会让显存和性能压力明显增加，建议 10 秒以内作为安全生产线。</span></div><div><b>二采说明</b><span>控制每秒有多少参考画面编码成视频特征；FPS 越高，外观变化、动作过程和时间连续性保留得越密，但编码更慢、显存占用更高。</span></div></div>`;
}

function emptyResultScaffoldMarkup({ staticPlaceholder = false } = {}) {
  const placeholderClass = staticPlaceholder ? " result-media-placeholder-static" : "";
  return `<div class="result-media-frame result-media-placeholder${placeholderClass}" aria-label="视频结果占位"><span>等待视频结果</span></div>
    <section id="usage-guide" class="usage-guide">${usageGuideMarkup()}</section>`;
}

function renderEmptyResultScaffold() {
  const result = $("result");
  if (!result || !result.classList.contains("empty-state")) return;
  result.className = "result-content result-empty";
  result.innerHTML = emptyResultScaffoldMarkup();
}

function renderResult(task) {
  const plan = task.plan || {};
  const result = task.result || {};
  const state = task.state;
  const status = statusLabels[state] || state;
  const error = result.error || task.message;
  const experiment = task.experimental || task.plan?.compiled?.advanced || {};
  const experimentReceipt = result.executionReceipt || task.plan?.executionReceipt || {};
  const diagnostics = JSON.stringify(result.runtimeStages ? result : plan, null, 2);
  const timingDetails = timingBreakdownMarkup(task);
  const historical = selectedTaskSource === "history" && activeTask && activeTask.id !== task.id;
  const title = state === "error" ? "后端错误" : state === "completed" ? (historical ? "历史结果" : "生成结果") : state === "dry_run_complete" ? "参数预览" : "实时执行";
  const historyHint = historical
    ? `<p class="hint">当前查看历史任务 ${escapeHtml(task.id)}；实时硬件状态属于活动任务 ${escapeHtml(activeTask.id)}。</p>`
    : "";
  const outputPath = taskOutputPath(task);
  const previewUrl = `api/output/${encodeURIComponent(task.id)}/h3_result.mp4`;
  const previewKey = `${task.id}:${previewUrl}`;
  const resultContainer = $("result");
  const terminalRenderKey = `${task.id}:${state}:${error || ""}:${result.failedStage || ""}`;
  if (terminalStates.has(state) && resultContainer.dataset.terminalRenderKey === terminalRenderKey) {
    renderEnhancementCard(task);
    return;
  }
  if (terminalStates.has(state)) resultContainer.dataset.terminalRenderKey = terminalRenderKey;
  else delete resultContainer.dataset.terminalRenderKey;
  const liveRenderKey = `${task.id}:${state}`;
  if (!terminalStates.has(state)
    && !result.outputAuthentic
    && resultContainer.dataset.liveRenderKey === liveRenderKey
    && resultContainer.querySelector(".result-media-placeholder")) {
    return;
  }
  if (!terminalStates.has(state)) resultContainer.dataset.liveRenderKey = liveRenderKey;
  else delete resultContainer.dataset.liveRenderKey;
  const stateChanged = resultContainer.dataset.visualState !== state;
  resultContainer.dataset.visualState = state;
  const existingVideo = resultContainer.querySelector("video.result-video");
  if (result.outputAuthentic && outputPath && existingVideo && existingVideo.dataset.previewKey === previewKey) {
    renderEnhancementCard(task);
    renderNvidiaVsrStatus(task);
    return;
  }
  const body = state === "error"
    ? `<p class="error">${escapeHtml(userErrorMessage(error))}</p><p class="hint">失败阶段：${escapeHtml(result.failedStage || "未细分")}</p>`
    : state === "completed"
      ? `${historyHint}<p class="hint">真实推理已完成：${result.outputAuthentic ? "已通过真实采样、解码和封装" : "未生成可验证成片"}</p>${outputPath ? `<p class="hint">输出：${escapeHtml(outputPath)}</p>` : ""}`
      : `<p class="hint">${escapeHtml(task.message || "后端正在执行")}</p>`;
  resultContainer.className = "result-content";
  resultContainer.innerHTML = `<div class="queue-meta"><strong>${title}</strong><span class="state">${escapeHtml(status)}</span></div>${body}${timingDetails}<details class="diagnostic-details"><summary>展开执行诊断</summary><pre class="plan-view">${escapeHtml(diagnostics)}</pre></details>`;
  if (experiment.experimentId) {
    const failure = result.error || experimentReceipt.failureReason || "";
    resultContainer.insertAdjacentHTML("afterbegin", `<p class="hint"><strong>隔离 KJ 实验</strong> · ${escapeHtml(experiment.experimentId)} · commit 60cd6bc · ${escapeHtml(state)}${failure ? ` · 失败原因：${escapeHtml(failure)}` : ""}</p>`);
  }
  if (!result.outputAuthentic || !outputPath) {
    resultContainer.insertAdjacentHTML("beforeend", emptyResultScaffoldMarkup({ staticPlaceholder: state === "error" || state === "cancelled" }));
  }
  const resultError = resultContainer.querySelector(".error");
  if (resultError) {
    const fullError = resultError.textContent;
    const overflowed = observeFitText(resultError, { minSize: 10, maxSize: 12 });
    if (overflowed) {
      const details = document.createElement("details");
      details.className = "result-error-details";
      details.innerHTML = "<summary>查看完整错误</summary><p></p>";
      details.querySelector("p").textContent = fullError;
      resultError.textContent = compactVisibleText(fullError, 260);
      resultError.insertAdjacentElement("afterend", details);
    }
  }
  if (!["dry_run_complete", "error"].includes(state)) {
    const diagnosticDetails = resultContainer.querySelector(".diagnostic-details");
    if (diagnosticDetails) diagnosticDetails.remove();
  }
  if (result.outputAuthentic && outputPath) {
    const video = document.createElement("video");
    video.className = "result-video";
    video.controls = true;
    video.preload = "metadata";
    video.src = previewUrl;
    video.dataset.previewKey = previewKey;
    const frame = document.createElement("div");
    frame.className = "result-media-frame";
    const canvas = task.compiled?.canvas?.export || task.plan?.canvas?.export || result.canvas?.export || null;
    const outputWidth = Number(canvas?.width || result.outputWidth || task.outputWidth);
    const outputHeight = Number(canvas?.height || result.outputHeight || task.outputHeight);
    if (Number.isFinite(outputWidth) && Number.isFinite(outputHeight) && outputWidth > 0 && outputHeight > 0) {
      video.style.aspectRatio = `${outputWidth} / ${outputHeight}`;
      frame.classList.toggle("is-tall", outputHeight > outputWidth);
    }
    frame.append(video);
    const previewNote = document.createElement("p");
    previewNote.className = "result-preview-note";
    previewNote.textContent = "生成结果 · 原始比例预览";
    const details = resultContainer.querySelector(".diagnostic-details");
    resultContainer.insertBefore(previewNote, details || null);
    resultContainer.insertBefore(frame, details || null);
  }
  renderEnhancementCard(task);
  renderNvidiaVsrStatus(task);
  if (stateChanged) replayVisualState(resultContainer, state === "completed" ? "result-arrived" : "state-transition");
}

let selectedEnhancement = null;
let enhancementTasks = [];
let enhancementPollTimer = null;
const enhancementLookupRequested = new Set();
const FLASHVSR_AVAILABLE = false;

function enhancementPreviewUrl(task) {
  const parentId = task?.parentTaskId || task?.sourceTaskId || task?.id;
  const file = task?.outputFile || String(task?.outputPath || "").split("/").pop();
  return `api/output/${encodeURIComponent(parentId)}/${encodeURIComponent(file)}`;
}

function latestEnhancementForSource(tasks, sourceTaskId) {
  return (tasks || [])
    .filter((item) => item.sourceTaskId === sourceTaskId)
    .sort((a, b) => (parseTaskTime(b.createdAt || b.updatedAt) || 0) - (parseTaskTime(a.createdAt || a.updatedAt) || 0))[0] || null;
}

function resultHistoryLabel(item) {
  const timestamp = parseTaskTime(item?.createdAt);
  const time = timestamp === null ? "时间未知" : new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false,
  }).format(timestamp);
  const resolution = item?.resolution?.label || item?.resolution?.id
    || (item?.resolution?.width && item?.resolution?.height ? `${item.resolution.width}×${item.resolution.height}` : "规格未知");
  const sourceId = String(item?.sourceTaskId || item?.taskId || "");
  return `${item?.kind === "repair" ? "修复" : "生成"} · ${time} · ${resolution} · ${sourceId.slice(-6)}`;
}

async function refreshResultHistory() {
  if (resultHistoryRequest) return resultHistoryRequest;
  resultHistoryRequest = Promise.all([
    fetch("api/tasks").then((response) => response.ok ? response.json() : Promise.reject(new Error("task history unavailable"))),
    fetch("api/postprocess/tasks").then((response) => response.ok ? response.json() : Promise.reject(new Error("postprocess history unavailable"))),
    fetch("api/nvidia-vsr/tasks").then((response) => response.ok ? response.json() : Promise.reject(new Error("nvidia vsr history unavailable"))),
  ])
    .then(([taskPayload, postprocessPayload, nvidiaVsrPayload]) => {
      const taskResults = (Array.isArray(taskPayload?.tasks) ? taskPayload.tasks : taskPayload || [])
        .filter((task) => task?.state === "completed" && task?.result?.outputAuthentic === true)
        .map((task) => {
          const receipt = task.result?.executionReceipt || task.plan?.executionReceipt || {};
          return {
            id: `generation:${task.id}`,
            kind: "generation",
            taskId: task.id,
            sourceTaskId: task.id,
            createdAt: task.completedAt || task.createdAt,
            resolution: receipt.effectiveResolution || {},
            url: `api/output/${encodeURIComponent(task.id)}/h3_result.mp4`,
            outputAuthentic: true,
          };
        });
      const postprocessResults = (Array.isArray(postprocessPayload?.tasks) ? postprocessPayload.tasks : [])
        .filter((task) => task?.outputAuthentic === true)
        .map((task) => ({
          ...task,
          kind: "repair",
          url: enhancementPreviewUrl(task),
          resolution: task.targetResolution || {},
        }));
      const nvidiaTasks = Array.isArray(nvidiaVsrPayload?.tasks) ? nvidiaVsrPayload.tasks : [];
      nvidiaTasks.forEach((task) => nvidiaVsrTasks.set(task.sourceTaskId || task.parentTaskId, task));
      const nextResults = [...taskResults, ...postprocessResults]
        .sort((a, b) => (parseTaskTime(b.createdAt || b.updatedAt) || 0) - (parseTaskTime(a.createdAt || a.updatedAt) || 0));
      const newlyDiscovered = resultHistoryLoaded
        ? nextResults.filter((item) => !observedResultIds.has(item.id))
        : [];
      nextResults.forEach((item) => observedResultIds.add(item.id));
      resultHistory = nextResults;
      if (newlyDiscovered.length) pendingLatestResult = newlyDiscovered[0];
      resultHistoryLoaded = true;
      return resultHistory;
    })
    .catch(() => resultHistory)
    .finally(() => { resultHistoryRequest = null; });
  return resultHistoryRequest;
}

function resultHistoryForTask(sourceTask) {
  const currentResultId = `generation:${sourceTask?.id || ""}`;
  const hasCurrentResult = sourceTask?.state === "completed"
    && sourceTask?.result?.outputAuthentic === true
    && Boolean(taskOutputPath(sourceTask));
  if (!hasCurrentResult || resultHistory.some((item) => item.id === currentResultId)) return resultHistory;
  const resolution = executionReceipt(sourceTask)?.effectiveResolution || {};
  return [{
    id: currentResultId,
    kind: "generation",
    taskId: sourceTask.id,
    sourceTaskId: sourceTask.id,
    createdAt: sourceTask.completedAt || sourceTask.createdAt,
    resolution: { id: resolution.preset, label: resolution.preset, width: resolution.width, height: resolution.height },
    url: `api/output/${encodeURIComponent(sourceTask.id)}/h3_result.mp4`,
    outputAuthentic: true,
  }, ...resultHistory];
}

function applyResultPreview(item, fallbackUrl) {
  const video = document.querySelector("video.result-video");
  const previewUrl = item?.url || fallbackUrl;
  if (video && previewUrl && video.getAttribute("src") !== previewUrl) {
    video.src = previewUrl;
    video.load();
  }
  const note = document.querySelector(".result-preview-note");
  if (note && item) note.textContent = resultHistoryLabel(item);
}

async function selectHistoryResult(sourceTask, item, fallbackUrl) {
  if (!item) return;
  selectedResultId = item.id;
  applyResultPreview(item, fallbackUrl);
  if (!item.sourceTaskId || item.sourceTaskId === sourceTask.id) return;
  const detail = await fetchTaskDetail({ id: item.sourceTaskId });
  if (detail?.id === item.sourceTaskId) watchTask(detail, { source: "history" });
}

function enhancementElapsed(task, now = Date.now()) {
  const startedAt = parseTaskTime(task?.acceptedAt || task?.createdAt);
  if (startedAt === null) return null;
  const finished = postprocessTerminalStates.has(task?.state);
  const finishedAt = finished ? parseTaskTime(task?.completedAt || task?.updatedAt) : null;
  if (finished && finishedAt === null) return null;
  return formatElapsed(Math.max(0, (finishedAt ?? now) - startedAt));
}

function enhancementHeartbeatAge(task, now = Date.now()) {
  const observedAt = parseTaskTime(task?.runtimeHeartbeat?.observedAt);
  return observedAt === null ? null : formatElapsed(Math.max(0, now - observedAt));
}

function renderEnhancementCard(sourceTask) {
  if (sourceTask?.state !== "completed" || sourceTask?.result?.outputAuthentic !== true || !taskOutputPath(sourceTask)) return;
  const resultContainer = $("result");
  let card = $("usage-guide");
  if (!card) {
    card = document.createElement("section");
    card.id = "usage-guide";
    card.className = "usage-guide";
    resultContainer.append(card);
  }
  if (!card.dataset.ready) {
    card.innerHTML = usageGuideMarkup();
    card.dataset.ready = "true";
  }
}

async function cancelEnhancement(sourceTask, post, button) {
  if (!post || postprocessTerminalStates.has(post.state) || button.disabled) return;
  button.disabled = true;
  button.textContent = "正在取消";
  try {
    const response = await fetch(`api/postprocess/tasks/${encodeURIComponent(post.id)}/cancel`, { method: "POST" });
    const task = await response.json();
    if (!response.ok) throw new Error(task.error || "取消修复失败");
    selectedEnhancement = task;
    enhancementTasks = [task, ...enhancementTasks.filter((item) => item.id !== task.id)];
    renderEnhancementCard(sourceTask);
  } catch (error) {
    button.disabled = false;
    button.textContent = "取消修复";
    setError(error.message || "取消修复失败");
  }
}

async function createEnhancement(sourceTask, target, selectedResult) {
  if (target !== "1080p") return;
  const buttons = [...document.querySelectorAll("#flashvsr-enhance [data-repair-target]")];
  buttons.forEach((button) => { button.disabled = true; });
  try {
    const response = await fetch(`api/tasks/${encodeURIComponent(sourceTask.id)}/enhance`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ targetResolution: target, sourceResultId: selectedResult?.id || `generation:${sourceTask.id}` }),
    });
    const task = await response.json();
    if (!response.ok) throw new Error(task.error || "高清修复任务创建失败");
    selectedEnhancement = task;
    enhancementTasks = [task, ...enhancementTasks.filter((item) => item.id !== task.id)];
    watchEnhancement(sourceTask, task.id);
  } catch (error) {
    setError(error.message || "高清修复任务创建失败");
    buttons.forEach((button) => { button.disabled = false; });
  }
}

function watchEnhancement(sourceTask, taskId) {
  clearInterval(enhancementPollTimer);
  renderEnhancementCard(sourceTask);
  enhancementPollTimer = setInterval(async () => {
    const response = await fetch(`api/postprocess/tasks/${encodeURIComponent(taskId)}`);
    if (!response.ok) return;
    selectedEnhancement = await response.json();
    enhancementTasks = [selectedEnhancement, ...enhancementTasks.filter((item) => item.id !== selectedEnhancement.id)];
    renderEnhancementCard(sourceTask);
    if (postprocessTerminalStates.has(selectedEnhancement.state)) {
      clearInterval(enhancementPollTimer);
      if (selectedEnhancement.state === "completed" && selectedEnhancement.outputAuthentic === true) {
        const outputId = selectedEnhancement.targetResolution?.id || "1080p";
        selectedResultId = `repair:${selectedEnhancement.id}:${outputId}`;
        refreshResultHistory().then(() => renderEnhancementCard(sourceTask));
      }
    }
  }, 500);
}

function formatGiB(value) {
  return Number.isFinite(Number(value)) ? `${Number(value).toFixed(1)} GiB` : "不可用";
}

function formatPercent(value) {
  return Number.isFinite(Number(value)) ? `${Math.round(Number(value))}%` : "不可用";
}

function setHardwareText(selector, value, warm = false) {
  const element = document.querySelector(selector);
  if (!element) return;
  element.textContent = value;
  element.classList.toggle("hardware-warm", warm);
}

function setHardwareBar(selector, percent) {
  const bar = document.querySelector(selector);
  if (bar) bar.style.width = `${Math.max(0, Math.min(100, Number(percent) || 0))}%`;
}

function hardwareTimeLabel(timestamp) {
  return timestamp ? new Intl.DateTimeFormat("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(timestamp) : "尚无数据";
}

function setHardwareConnection(label, state = "connecting") {
  const service = $("hardware-service");
  if (!service) return;
  service.className = `hardware-connection is-${state}`;
  service.innerHTML = `<i aria-hidden="true"></i><span>${escapeHtml(label)}</span>`;
}

function renderHardwareStatus(status) {
  const gpu = status.gpu || {};
  const cpu = status.cpu || {};
  const memory = status.memory || {};
  const runtime = status.runtime || {};
  const service = $("hardware-service");
  if (service) service.textContent = runtime.service === "ready" ? (runtime.taskState === "idle" ? "空闲" : "运行中") : "不可用";
  if (gpu.available) {
    const temperature = Number(gpu.temperatureC);
    setHardwareText("[data-hw-gpu-temperature]", Number.isFinite(temperature) ? `${temperature.toFixed(0)} °C` : "不可用", Number.isFinite(temperature) && temperature >= 80);
    setHardwareText("[data-hw-gpu-load]", `核心负载：${formatPercent(gpu.coreLoadPercent)}`);
    setHardwareBar("[data-hw-gpu-load-bar]", gpu.coreLoadPercent);
    setHardwareText("[data-hw-gpu-memory]", `显存：${formatGiB(Number(gpu.memoryUsedMiB) / 1024)} / ${formatGiB(Number(gpu.memoryTotalMiB) / 1024)}（${formatPercent(gpu.memoryPercent)}）`);
    setHardwareText("[data-hw-gpu-power]", `功耗：${Number.isFinite(Number(gpu.powerDrawW)) ? `${Number(gpu.powerDrawW).toFixed(0)} W` : "不可用"} / ${Number.isFinite(Number(gpu.powerLimitW)) ? `${Number(gpu.powerLimitW).toFixed(0)} W` : "不可用"}`);
    setHardwareText("[data-hw-gpu-extra]", `风扇：${formatPercent(gpu.fanPercent)} · 编码/解码：${formatPercent(gpu.encoderLoadPercent)} / ${formatPercent(gpu.decoderLoadPercent)}`);
  } else {
    setHardwareText("[data-hw-gpu-temperature]", "GPU 不可用"); setHardwareText("[data-hw-gpu-load]", "核心负载：不可用"); setHardwareText("[data-hw-gpu-memory]", "显存：不可用"); setHardwareText("[data-hw-gpu-power]", "功耗：不可用"); setHardwareText("[data-hw-gpu-extra]", "风扇/编解码：不可用"); setHardwareBar("[data-hw-gpu-load-bar]", 0);
  }
  setHardwareText("[data-hw-cpu-load]", cpu.available ? `系统负载：${formatPercent(cpu.loadPercent)}` : "CPU 采样中");
  setHardwareText("[data-hw-cpu-cores]", `逻辑核心：${cpu.logicalCores || "不可用"}`); setHardwareBar("[data-hw-cpu-load-bar]", cpu.loadPercent);
  setHardwareText("[data-hw-worker]", runtime.workerAlive ? `H3 worker：PID ${runtime.workerPid}` : "H3 worker：无");
  setHardwareText("[data-hw-memory-used]", memory.available ? `已用：${formatGiB(memory.usedGiB)} / ${formatGiB(memory.totalGiB)}` : "内存不可用");
  setHardwareText("[data-hw-memory-available]", `可用：${formatGiB(memory.availableGiB)}`); setHardwareText("[data-hw-memory-percent]", memory.available ? `使用率：${formatPercent(memory.usedPercent)}` : "使用率：不可用"); setHardwareBar("[data-hw-memory-bar]", memory.usedPercent);
  setHardwareText("[data-hw-runtime]", runtime.taskState === "idle" ? "服务空闲" : `任务：${runtime.taskState || "未知"}`);
  setHardwareText("[data-hw-runtime-task]", runtime.taskId ? `活动任务：${runtime.taskId}` : "活动任务：无");
  setHardwareText("[data-hw-stage]", `当前阶段：${stageLabels[runtime.taskStage] || runtime.taskStage || "空闲"}`);
  setHardwareText("[data-hw-cache]", `采样：${status.cached ? "缓存" : "实时"}${Number.isFinite(Number(status.cacheAgeMilliseconds)) ? ` · ${Math.round(Number(status.cacheAgeMilliseconds))} ms` : ""}`);
}

function removeLegacyCreateControls() {
  document.querySelector(".action-row")?.remove();
}

let hoverSelectId = 0;
let activeHoverSelect = null;

document.addEventListener("pointerdown", (event) => {
  if (activeHoverSelect && !event.target.closest?.(".hover-select")) {
    activeHoverSelect.close();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && activeHoverSelect) {
    activeHoverSelect.close();
  }
});

function installHoverSelect(selectTarget) {
  const select = typeof selectTarget === "string" ? $(selectTarget) : selectTarget;
  if (!select || select.dataset.hoverSelectInstalled === "true") return;
  select.dataset.hoverSelectInstalled = "true";
  select.classList.add("native-select-source");
  select.tabIndex = -1;
  select.setAttribute("aria-hidden", "true");

  const picker = document.createElement("div");
  picker.className = "hover-select";
  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "hover-select-trigger";
  trigger.setAttribute("aria-haspopup", "listbox");
  trigger.setAttribute("aria-expanded", "false");
  const menu = document.createElement("div");
  menu.className = "hover-select-menu";
  const controlId = select.id || `hover-select-${++hoverSelectId}`;
  menu.id = `${controlId}-hover-menu`;
  menu.setAttribute("role", "listbox");
  menu.hidden = true;
  trigger.setAttribute("aria-controls", menu.id);
  let closeTimer = null;

  const optionButtons = Array.from(select.options).map((option) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "hover-select-option";
    button.dataset.value = option.value;
    button.textContent = option.textContent;
    button.setAttribute("role", "option");
    button.addEventListener("click", () => {
      if (select.value !== option.value) {
        select.value = option.value;
        select.dispatchEvent(new Event("change", { bubbles: true }));
      }
      saveComposerDraft();
      refresh();
      setOpen(false);
      trigger.focus();
    });
    menu.append(button);
    return button;
  });

  function refresh() {
    const selected = select.options[select.selectedIndex];
    trigger.textContent = selected?.textContent || "请选择";
    trigger.disabled = select.disabled;
    trigger.setAttribute("aria-disabled", String(select.disabled));
    optionButtons.forEach((button) => {
      const active = button.dataset.value === select.value;
      button.classList.toggle("is-selected", active);
      button.setAttribute("aria-selected", String(active));
    });
  }

  function setOpen(open) {
    if (closeTimer) {
      clearTimeout(closeTimer);
      closeTimer = null;
    }
    open = Boolean(open && !select.disabled);
    if (open && activeHoverSelect && activeHoverSelect !== picker) {
      activeHoverSelect.close();
    }
    picker.classList.toggle("is-open", open);
    menu.hidden = !open;
    trigger.setAttribute("aria-expanded", String(open));
    if (!open) delete picker.dataset.pinned;
    if (open) activeHoverSelect = picker;
    else if (activeHoverSelect === picker) activeHoverSelect = null;
  }

  function scheduleClose() {
    if (picker.dataset.pinned === "true") return;
    closeTimer = setTimeout(() => setOpen(false), 180);
  }

  picker.addEventListener("mouseenter", () => setOpen(true));
  picker.addEventListener("mouseleave", scheduleClose);
  picker.close = () => setOpen(false);
  trigger.addEventListener("click", () => {
    const open = menu.hidden;
    setOpen(open);
    if (open) picker.dataset.pinned = "true";
  });
  trigger.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown" || event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      setOpen(true);
      optionButtons[Math.max(0, select.selectedIndex)]?.focus();
    } else if (event.key === "Escape") {
      setOpen(false);
    }
  });
  menu.addEventListener("keydown", (event) => {
    const current = optionButtons.indexOf(document.activeElement);
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const direction = event.key === "ArrowDown" ? 1 : -1;
      optionButtons[(current + direction + optionButtons.length) % optionButtons.length]?.focus();
    } else if (event.key === "Escape") {
      setOpen(false);
      trigger.focus();
    }
  });
  select.addEventListener("change", refresh);
  picker.append(trigger, menu);
  select.insertAdjacentElement("afterend", picker);
  refresh();
}

function arrangeGenerationActionRow() {
  const mode = $("mode");
  const duration = $("duration");
  const generate = $("generate");
  const settingsGrid = document.querySelector(".basic-settings-grid");
  const row = $("generation-actions");
  if (!mode || !duration || !generate || !settingsGrid || !row) return;
  const modeField = mode.parentElement;
  const durationField = duration.parentElement;
  const modeGrid = modeField && modeField.parentElement;
    settingsGrid.prepend(modeField);
    settingsGrid.append(durationField);
  const controlRow = row.querySelector(".kernel-control-row");
  if (controlRow && generate.parentElement !== controlRow) controlRow.append(generate);
  else if (!controlRow) row.append(generate);
  if (modeGrid && modeGrid.classList.contains("mode-grid")) modeGrid.remove();
}

async function refreshHardwareStatus() {
  if (hardwareRequestInFlight) return;
  hardwareRequestInFlight = true;
  try {
    const response = await fetch("api/hardware-status");
    if (!response.ok) throw new Error(response.status === 404 ? "hardware_endpoint_missing" : `hardware_status_${response.status}`);
    const status = await response.json();
    lastHardwareStatus = status;
    lastHardwareUpdatedAt = new Date();
    observeServiceLifecycle(true);
    renderHardwareStatus(status);
    const runtime = status.runtime || {};
    const state = runtime.service === "ready" ? "live" : "stale";
    const label = runtime.service === "ready"
      ? `实时 · 更新 ${hardwareTimeLabel(lastHardwareUpdatedAt)}`
      : `服务状态不可用 · 更新 ${hardwareTimeLabel(lastHardwareUpdatedAt)}`;
    setHardwareConnection(label, state);
  } catch (error) {
    observeServiceLifecycle(false);
    if (lastHardwareStatus) {
      setHardwareConnection(`数据已过期 · 上次更新 ${hardwareTimeLabel(lastHardwareUpdatedAt)}`, "stale");
    } else {
      renderHardwareStatus({ runtime: { service: "unavailable", taskState: "unavailable", taskStage: "unavailable" } });
      const message = error && error.message === "hardware_endpoint_missing"
        ? "当前服务尚未提供硬件状态，请重启 H3 服务后生效"
        : "硬件状态暂不可用，正在自动重试";
      setHardwareConnection(message, "error");
    }
  } finally {
    hardwareRequestInFlight = false;
  }
}

async function loadHistory() {
  try {
    await refreshResultHistory();
    const response = await fetch("api/tasks");
    if (!response.ok) return;
    const taskResponse = await response.json();
    const tasks = Array.isArray(taskResponse) ? taskResponse : (taskResponse.tasks || []);
    activeTask = latestTask(tasks, isActiveTask);
    let selectedTaskId = "";
    try { selectedTaskId = sessionStorage.getItem(SELECTED_TASK_STORAGE_KEY) || ""; } catch (_error) { /* storage unavailable */ }
    const restoredTask = (selectedTaskId && tasks.find((task) => task.id === selectedTaskId)) || activeTask;
    if (restoredTask) {
      const detail = await fetchTaskDetail(restoredTask);
      if (!manualComposerStatePresent) hydrateComposerFromTask(detail);
      watchTask(detail, { source: activeTask && activeTask.id === detail.id ? "active" : "history" });
    }
  } catch (_error) {
    // History is helpful but never blocks creating a new task.
  }
}

async function refreshTaskSelection() {
  try {
    await refreshResultHistory();
    const response = await fetch("api/tasks");
    if (!response.ok) return;
    const taskResponse = await response.json();
    const tasks = Array.isArray(taskResponse) ? taskResponse : (taskResponse.tasks || []);
    const nextActive = latestTask(tasks, isActiveTask);
    activeTask = nextActive;
    const latestResult = pendingLatestResult;
    if (latestResult) {
      const detail = await fetchTaskDetail({ id: latestResult.sourceTaskId });
      if (detail?.id === latestResult.sourceTaskId) {
        pendingLatestResult = null;
        selectedResultId = latestResult.id;
        watchTask(detail, { source: "latest-result" });
      }
    }
    if (!selectedTask && nextActive) {
      const detail = await fetchTaskDetail(nextActive);
      if (!manualComposerStatePresent) hydrateComposerFromTask(detail);
      watchTask(detail, { source: "active" });
    } else if (nextActive && selectedTaskSource === "active" && selectedTask?.id !== nextActive.id) {
      const detail = await fetchTaskDetail(nextActive);
      if (!manualComposerStatePresent) hydrateComposerFromTask(detail);
      watchTask(detail, { source: "active" });
    }
    if (selectedTask) {
      renderQueue(selectedTask, { advanceProgress: false });
      renderResult(selectedTask);
    }
  } catch (_error) {
    // Hardware polling and task selection are best-effort; the task poll owns execution truth.
  }
}

function watchTask(task, { source = "selected" } = {}) {
  const previousTask = selectedTask;
  if (source === "active" && previousTask?.id !== task.id) {
    selectedResultId = null;
    completionHistoryRetryAt = 0;
  }
  selectedTask = task;
  selectedTaskSource = source;
  try { sessionStorage.setItem(SELECTED_TASK_STORAGE_KEY, task.id); } catch (_error) { /* storage unavailable */ }
  if (source === "active" || isActiveTask(task)) activeTask = task;
  renderQueue(task);
  renderResult(task);
  clearInterval(pollTimer);
  const terminal = ["dry_run_complete", "completed", "error", "cancelled"];
  if (terminal.includes(task.state)) {
    $("generate").disabled = false;
    return;
  }
  $("generate").disabled = true;
  pollTimer = setInterval(async () => {
    if (!selectedTask?.id || selectedTask.id !== task.id) return;
    try {
      const response = await fetch(`api/tasks/${encodeURIComponent(task.id)}`, { cache: "no-store" });
      if (!response.ok) return;
      const next = await response.json();
      if (!next?.id || next.id !== task.id || selectedTask?.id !== task.id) return;
      const completedNow = next.state === "completed" && selectedTask.state !== "completed";
      selectedTask = next;
      if (isActiveTask(next) || next.state === "completed") activeTask = next;
      renderQueue(next, { advanceProgress: false });
      renderResult(next);
      if (completedNow && next.result?.outputAuthentic === true) {
        selectedResultId = `generation:${next.id}`;
        void refreshResultHistory();
      }
      if (terminalStates.has(next.state)) {
        clearInterval(pollTimer);
        pollTimer = null;
        $("generate").disabled = false;
      }
    } catch (_error) {
      // SSE remains the fast path; this poll is the authoritative fallback.
    }
  }, 1000);
}

function connectTaskEvents() {
  if (typeof EventSource === "undefined") return;
  taskEventSource?.close();
  taskEventSource = new EventSource("api/events");
  taskEventSource.addEventListener("task", (event) => {
    try {
      const next = JSON.parse(event.data);
      if (!next?.id) return;
      
      if (!selectedTask || selectedTask.id !== next.id) {
        if (isActiveTask(next)) watchTask(next, { source: "active" });
        return;
      }
      
      const completedNow = next.state === "completed" && selectedTask.state !== "completed";
      selectedTask = { ...selectedTask, ...next };
      if (isActiveTask(next) || next.state === "completed") activeTask = selectedTask;
       if (completedNow && next.result?.outputAuthentic === true) {
        selectedResultId = `generation:${next.id}`;
          void fetchTaskDetail(next).then((detail) => {
            if (!detail?.id) throw new Error("生成完成回执缺少任务详情");
            selectedTask = { ...selectedTask, ...detail };
            renderResult(selectedTask);
            return refreshResultHistory().then(() => startNvidiaVsr(selectedTask));
          }).catch((error) => setError(error.message));
      }
      renderQueue(selectedTask);
      renderResult(selectedTask);
      $("generate").disabled = !terminalStates.has(selectedTask.state);
    } catch (_error) { /* EventSource reconnects automatically. */ }
  });
  taskEventSource.addEventListener("hardware", (event) => {
    try {
      const status = JSON.parse(event.data);
      lastHardwareStatus = status;
      lastHardwareUpdatedAt = new Date();
      renderHardwareStatus(status);
      setHardwareConnection(`实时 · 更新 ${hardwareTimeLabel(lastHardwareUpdatedAt)}`, "live");
    } catch (_error) { /* Ignore one malformed frame. */ }
  });
}

async function createTask() {
  setError("");
  $("generate").disabled = true;
  try {
    const mode = $("mode");
    syncGenerationModeFromReferences();
    const requestPayload = payload();
    if (mode.value !== requestPayload.mode) {
      throw new Error("生成模式状态不同步，任务未提交，请重新选择生成模式后再试。");
    }
    const response = await fetch("api/tasks", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(requestPayload) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "任务创建失败");
    watchTask(data, { source: "active" });
  } catch (error) {
    setError(error.message);
    $("generate").disabled = false;
  }
}

async function cancelTaskLegacy() {
  if (!selectedTask) return;
  const button = $("cancel-task");
  if (button) { button.disabled = true; button.textContent = "正在停止"; }
  try {
    const response = await fetch(`api/tasks/${selectedTask.id}/cancel`, { method: "POST" });
    const task = await response.json();
    if (!response.ok) throw new Error(task.error || "停止生成失败");
    renderQueue(task); renderResult(task);
    // ``cancelling`` continues polling until the runner has released GPU
    // state and returned the terminal ``cancelled`` receipt.
    if (task.state === "cancelled") $("generate").disabled = false;
  } catch (error) {
    setError(error.message || "停止生成失败");
    if (button) { button.disabled = false; button.textContent = "停止生成"; }
  }
}

async function cancelTask() {
  if (!selectedTask || !isActiveTask(selectedTask)) return;
  const taskId = selectedTask.id;
  if (cancelPendingTaskIds.has(taskId)) return;
  cancelPendingTaskIds.add(taskId);
  renderQueue(selectedTask);
  try {
    const response = await fetch(`api/tasks/${taskId}/cancel`, { method: "POST" });
    const task = await response.json();
    if (!response.ok) throw new Error(task.error || "停止生成失败");
    renderQueue(task);
    renderResult(task);
    // `cancelling` remains a real intermediate state until the worker reaches
    // its next safe boundary and writes a terminal receipt.
    if (terminalStates.has(task.state)) $("generate").disabled = false;
  } catch (error) {
    cancelPendingTaskIds.delete(taskId);
    setError(error.message || "停止生成失败");
    if (selectedTask && selectedTask.id === taskId) renderQueue(selectedTask);
  }
}

$("file-input").addEventListener("change", async (event) => {
  const files = [...event.target.files];
  for (const file of files) {
    const kind = detectKind(file);
    if (!kind) { setError(`${file.name} 不是支持的图片、视频或音频文件`); continue; }
    if (refs.length >= MAX_REFERENCES) {
      setError("最多添加 15 个参考素材（图片9、视频3、音频3）");
      break;
    }
    const sameKindCount = refs.filter((ref) => ref.kind === kind).length;
    if (sameKindCount >= REFERENCE_LIMITS[kind]) {
      const labels = { image: "图片", video: "视频", audio: "独立音频" };
      setError(`最多添加 ${REFERENCE_LIMITS[kind]} 个${labels[kind]}；视频内含原声不另占独立音频名额`);
      continue;
    }
    try {
      const form = new FormData();
      form.append("file", file, file.name);
      const upload = await fetch("api/assets", { method: "POST", body: form });
      const data = await upload.json();
      if (!upload.ok) throw new Error(data.error || "资产上传到 input 失败");
      const previewUrl = kind === "image" ? URL.createObjectURL(file) : null;
      const actualKind = data.asset.kind;
      if (!REFERENCE_LIMITS[actualKind]) throw new Error("服务端未能识别素材类型");
      if (refs.filter((ref) => ref.kind === actualKind).length >= REFERENCE_LIMITS[actualKind]) throw new Error("该类素材已达到官方数量上限");
      refs.push({ ...data.asset, role: $("role").value, previewUrl });
    } catch (error) {
      setError(error.message);
    }
  }
  event.target.value = "";
  renderReferences();
  saveComposerDraft();
});
function installWheelSelect(id, min, max) {
  const select = $(id);
  if (!select || select.dataset.wheelSelectInstalled === "true") return;
  select.dataset.wheelSelectInstalled = "true";
  select.addEventListener("mouseenter", () => { select.dataset.wheelHover = "true"; });
  select.addEventListener("mouseleave", () => { select.dataset.wheelHover = "false"; });
}

document.addEventListener("wheel", (event) => {
  const select = event.target instanceof Element ? event.target.closest("select[data-wheel-select-installed='true']") : null;
  if (!select || select.dataset.wheelHover !== "true") return;
  event.preventDefault();
  const min = select.id === "duration" ? 4 : 1;
  const max = select.id === "duration" ? 15 : select.id === "reference-video-fps" ? 2 : 24;
  const current = Number(select.value || min);
  const next = select.id === "reference-video-vae-fps"
    ? (current === 12 ? 24 : 12)
    : Math.max(min, Math.min(max, current + (event.deltaY < 0 ? 1 : -1)));
  if (next === current) return;
  select.value = String(next);
  select.dispatchEvent(new Event("change", { bubbles: true }));
}, { capture: true, passive: false });

installWheelSelect("duration", 4, 15);
installWheelSelect("reference-video-fps", 1, 2);
installWheelSelect("reference-video-vae-fps", 12, 24);

$("mode").addEventListener("change", () => {
  setModeFeedback("");
  syncGenerationModeFromReferences();
  renderReferenceCapacity();
});
window.addEventListener("beforeunload", () => refs.forEach(releaseReferencePreview));
const compileButton = $("compile");
if (compileButton) compileButton.addEventListener("click", async () => { try { await compile(); } catch (error) { setError(error.message); } });

function installGenerateButtonMotion() {
  const button = $("generate");
  if (!button || !window.matchMedia("(hover: hover) and (pointer: fine)").matches || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const channels = {
    x: { value: 0, velocity: 0, target: 0, stiffness: 290, damping: 24 },
    y: { value: 0, velocity: 0, target: 0, stiffness: 290, damping: 24 },
    scaleX: { value: 1, velocity: 0, target: 1, stiffness: 340, damping: 27 },
    scaleY: { value: 1, velocity: 0, target: 1, stiffness: 340, damping: 27 },
    labelX: { value: 0, velocity: 0, target: 0, stiffness: 360, damping: 28 },
    labelY: { value: 0, velocity: 0, target: 0, stiffness: 360, damping: 28 },
  };
  let frame = 0;
  let lastTime = 0;
  let hovering = false;
  let pressing = false;
  let pointerX = 0;
  let pointerY = 0;

  const writeFrame = () => {
    button.style.setProperty("--generate-x", `${channels.x.value.toFixed(3)}px`);
    button.style.setProperty("--generate-y", `${channels.y.value.toFixed(3)}px`);
    button.style.setProperty("--generate-scale-x", channels.scaleX.value.toFixed(5));
    button.style.setProperty("--generate-scale-y", channels.scaleY.value.toFixed(5));
    button.style.setProperty("--generate-label-x", `${channels.labelX.value.toFixed(3)}px`);
    button.style.setProperty("--generate-label-y", `${channels.labelY.value.toFixed(3)}px`);
  };
  const settled = () => Object.values(channels).every((channel) =>
    Math.abs(channel.target - channel.value) < .0008 && Math.abs(channel.velocity) < .008
  );
  const tick = (time) => {
    const dt = Math.min((time - (lastTime || time)) / 1000, 1 / 30);
    lastTime = time;
    Object.values(channels).forEach((channel) => {
      const force = (channel.target - channel.value) * channel.stiffness - channel.velocity * channel.damping;
      channel.velocity += force * dt;
      channel.value += channel.velocity * dt;
    });
    writeFrame();
    if (settled()) {
      Object.values(channels).forEach((channel) => {
        channel.value = channel.target;
        channel.velocity = 0;
      });
      writeFrame();
      frame = 0;
      lastTime = 0;
      if (!hovering && !pressing) button.classList.remove("is-generate-active");
      return;
    }
    frame = requestAnimationFrame(tick);
  };
  const animate = () => {
    button.classList.add("is-generate-active");
    if (!frame) frame = requestAnimationFrame(tick);
  };
  const setTargets = () => {
    if (button.disabled || (!hovering && !pressing)) {
      channels.x.target = 0;
      channels.y.target = 0;
      channels.scaleX.target = 1;
      channels.scaleY.target = 1;
      channels.labelX.target = 0;
      channels.labelY.target = 0;
      animate();
      return;
    }
    const rect = button.getBoundingClientRect();
    const nx = Math.max(-1, Math.min(1, ((pointerX - rect.left) / rect.width - .5) * 2));
    const ny = Math.max(-1, Math.min(1, ((pointerY - rect.top) / rect.height - .5) * 2));
    channels.x.target = pressing ? nx * 1.5 : nx * 3.2;
    channels.y.target = pressing ? ny * .7 + .6 : ny * 1.6;
    channels.scaleX.target = pressing ? .994 : 1.004;
    channels.scaleY.target = pressing ? .955 : 1.038;
    channels.labelX.target = pressing ? nx * 1.8 : nx * 4.6;
    channels.labelY.target = pressing ? ny * .8 + .4 : ny * 2.1;
    animate();
  };
  const resetImmediately = () => {
    if (frame) cancelAnimationFrame(frame);
    frame = 0;
    lastTime = 0;
    hovering = false;
    pressing = false;
    Object.values(channels).forEach((channel) => {
      channel.target = channel === channels.scaleX || channel === channels.scaleY ? 1 : 0;
      channel.value = channel.target;
      channel.velocity = 0;
    });
    writeFrame();
    button.classList.remove("is-generate-active");
  };

  button.addEventListener("pointerenter", (event) => {
    if (button.disabled) return;
    hovering = true;
    pointerX = event.clientX;
    pointerY = event.clientY;
    setTargets();
  });
  button.addEventListener("pointermove", (event) => {
    if (button.disabled) return;
    pointerX = event.clientX;
    pointerY = event.clientY;
    setTargets();
  });
  button.addEventListener("pointerleave", () => {
    hovering = false;
    pressing = false;
    setTargets();
  });
  button.addEventListener("pointerdown", (event) => {
    if (button.disabled) return;
    pressing = true;
    pointerX = event.clientX;
    pointerY = event.clientY;
    setTargets();
  });
  window.addEventListener("pointerup", () => {
    if (!pressing) return;
    pressing = false;
    setTargets();
  });
  new MutationObserver(() => {
    if (button.disabled) resetImmediately();
  }).observe(button, { attributes: true, attributeFilter: ["disabled"] });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) resetImmediately();
  });
}

installGenerateButtonMotion();
$("generate").addEventListener("click", createTask);
removeLegacyCreateControls(); arrangeGenerationActionRow();
installReferencePolicyControl();
document.querySelectorAll("select[id]").forEach(installHoverSelect);
captureComposerDefaults();

async function initializeWorkbench() {
  restoreComposerForCurrentSession();
  renderReferences(); renderEmptyResultScaffold(); refreshBackendStatus(); refreshHardwareStatus();
  loadHistory();
}

initializeWorkbench();
installStatusPanelAlignment();
requestAnimationFrame(() => document.body.classList.add("ui-ready"));
document.addEventListener("input", (event) => {
  if (event.target?.matches?.("input[id]:not(#file-input), select[id], textarea[id], .video-segment")) saveComposerDraft();
});
document.addEventListener("change", (event) => {
  if (event.target?.matches?.("input[id]:not(#file-input), select[id], textarea[id], .video-segment")) saveComposerDraft();
});
document.addEventListener("click", (event) => {
  if (event.target?.closest?.(".hover-select-option, input[type='checkbox'], input[type='radio']")) {
    queueMicrotask(saveComposerDraft);
  }
});
setInterval(refreshTaskClock, 1000);
setInterval(refreshHardwareStatus, 1500);
connectTaskEvents();

function installStatusPanelAlignment() {
  const column = document.querySelector(".side-column");
  const queuePanel = column?.querySelector(".queue-panel");
  const queue = queuePanel?.querySelector("#queue");
  const heading = queuePanel?.querySelector(":scope > .section-heading");
  if (!column || !queuePanel || !queue || !heading) return;
  let frame = 0;
  const align = () => {
    frame = 0;
    if (!window.matchMedia("(min-width:901px)").matches) {
      column.style.removeProperty("--status-panel-height");
      return;
    }
    const style = getComputedStyle(queuePanel);
    const rowGap = parseFloat(style.rowGap);
    const height = Math.ceil(
      queue.scrollHeight
      + heading.offsetHeight
      + parseFloat(style.paddingTop)
      + parseFloat(style.paddingBottom)
      + (Number.isFinite(rowGap) ? rowGap : 0)
    );
    column.style.setProperty("--status-panel-height", `${Math.max(354, height)}px`);
  };
  const schedule = () => { cancelAnimationFrame(frame); frame = requestAnimationFrame(align); };
  new MutationObserver(schedule).observe(queue, { childList:true, subtree:true, characterData:true });
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(schedule).observe(queue);
  window.addEventListener("resize", schedule);
  align();
}
