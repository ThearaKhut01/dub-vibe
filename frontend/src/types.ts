export type ProjectMode = 'EDIT' | 'DUB';
export type ProjectStatus = 'Ready' | 'Processing' | 'Queued' | 'Exporting';
export type ProviderMode = 'LOCAL AI' | 'FREE ONLINE' | 'PAID API' | 'CUSTOM API';

export interface ProviderStatus {
  id: string;
  name: string;
  mode: ProviderMode;
  connected: boolean;
  primary: boolean;
  fallback: boolean;
}

export interface CustomApiConfig {
  base_url: string;
  /** The full chain as typed, e.g. "model-a, model-b". */
  model: string;
  /** The chain in fallback order. */
  models: string[];
  /** First model in the chain, used for labels. */
  primary: string;
  configured: boolean;
  has_key: boolean;
}

export interface TimelineSegment {
  id: string;
  speaker: string;
  start: number;
  end: number;
  text: string;
  translation: string;
  voice: string;
  color: string;
  /** Backend file path of the synthesized dub audio (dub segments only). */
  path?: string;
  /** Speaker gender chosen for this line's voice in the Dubbing Track editor. */
  gender?: 'Male' | 'Female';
  /** Generated clip length compared with its time slot (set after audio generation). */
  fit?: { clip_seconds: number | null; slot_seconds: number; overrun_seconds: number; fits: boolean };
}

export interface WorkflowStep {
  id: string;
  label: string;
  detail: string;
  status: 'Completed' | 'Active' | 'Queued' | 'Failed' | 'Skipped';
  progress: number;
}

export interface MediaAsset {
  id: string;
  name: string;
  type: 'Video' | 'Audio' | 'Subtitle';
  duration: string;
  size: string;
  resolution: string;
  badge: string;
}

export interface ProjectSummary {
  id: string;
  name: string;
  mode: ProjectMode;
  duration: string;
  status: ProjectStatus;
  progress: number;
  hasMedia: boolean;
  mediaName: string | null;
  mediaResolution: string | null;
  mediaSize: number | null;
}

export type ModelStatus = 'not_installed' | 'downloading' | 'validating' | 'installed' | 'loaded' | 'error';

export interface WhisperModelInfo {
  id: string;
  name: string;
  repo: string;
  installed: boolean;
  loaded: boolean;
  selected: boolean;
  status: ModelStatus;
  progress: number;
  downloaded_mb: number;
  total_mb: number;
  size_mb: number | null;
  path: string | null;
  error: string | null;
  // Set only by the VoxCPM2 status: why the model is not usable (missing or incomplete files).
  verify_error?: string | null;
  // Set only by the Qwen status: which llama.cpp backend will actually run.
  backend?: 'cpu' | 'cuda' | 'none' | 'unknown';
  backend_label?: string;
  backend_reason?: string | null;
  cuda_backend_installed?: boolean;
  runtime_problem?: string | null;
}

export interface HardwareInfo {
  cpu: string;
  cpu_cores: number;
  ram_gb: number;
  cuda_available: boolean;
  gpu: string | null;
  vram_gb: number | null;
  device: string;
  compute: string;
  resolved_device: string;
  resolved_compute: string;
}

export interface CudaPackInfo {
  id: string;
  name: string;
  installed: boolean;
  status: 'unavailable' | 'available' | 'downloading' | 'installed' | 'error';
  progress: number;
  downloaded_mb: number;
  total_mb: number;
  error: string | null;
  size_mb: number | null;
  gpu: { name: string; vram_gb: number } | null;
}

export interface ModelStorageInfo {
  root: string;
  free_gb: number;
  total_gb: number;
  installed_mb: number;
  models: Record<string, number>;
}

export interface OverlapProposal {
  segment_id: string;
  type: 'speed_adjust' | 'silence_absorb' | 'needs_retranslation';
  old_start: number;
  old_end: number;
  old_speed?: number;
  new_start?: number;
  new_end?: number;
  new_speed?: number;
  new_audio_path?: string | null;
  actual_duration?: number;
  slot_duration?: number;
  needs_retranslation?: boolean;
  target_char_count?: number;
  description: string;
}

export interface FixOverlapsResponse {
  overlaps_found: number;
  proposals: OverlapProposal[];
}

// --- Multi-provider API pool ------------------------------------------------

export type ApiPoolProviderId = 'gemini' | 'openai' | 'codecraft' | 'custom';
export type ApiPoolStrategy = 'round_robin' | 'least_used' | 'hybrid';
export type ApiPoolHealthState = 'cooling' | 'degraded' | 'healthy' | 'unknown';

export interface ApiPoolUsageLimits {
  requests_today: number;
  requests_month: number;
  /** null = no limit configured; the pool never invents one. */
  daily_limit: number | null;
  monthly_limit: number | null;
  daily_remaining: number | null;
  monthly_remaining: number | null;
  allowed: boolean;
  reason: string | null;
}

export interface ApiPoolHealth {
  state: ApiPoolHealthState;
  cooldown_remaining: number;
  cooldown_reason: string;
  consecutive_failures: number;
  last_error: string;
  last_error_at: number | null;
  last_success_at: number | null;
  last_latency_ms: number;
  total_requests: number;
  total_failures: number;
}

export interface ApiPoolEntry {
  id: string;
  provider: ApiPoolProviderId;
  provider_label: string;
  model: string;
  label: string;
  display_name: string;
  base_url: string;
  enabled: boolean;
  weight: number;
  daily_limit: number | null;
  monthly_limit: number | null;
  key_source: 'pool' | 'legacy';
  has_key: boolean;
  /** Last four characters only — the key itself never leaves the backend. */
  key_hint: string;
  notes: string;
  created_at: number;
  health: ApiPoolHealth;
  usage: ApiPoolUsageLimits;
  usable: boolean;
  unavailable_reason: string;
}

export interface ApiPoolUsageTotals {
  requests: number;
  successes: number;
  failures: number;
  tokens_in: number;
  tokens_out: number;
  chars_in: number;
  chars_out: number;
}

export interface ApiPoolQueueItem {
  id: number;
  provider: string;
  entry_id: string | null;
  model: string | null;
  status: string;
  attempts: number;
  error: string | null;
  created_at: number;
  updated_at: number;
}

export interface ApiPoolStatus {
  settings: {
    strategy: ApiPoolStrategy;
    cache_enabled: boolean;
    cache_ttl_days: number;
    max_attempts: number;
    request_timeout: number;
    max_parallel: number;
    strategies: ApiPoolStrategy[];
  };
  providers: Record<string, {
    label: string;
    entries: number;
    usable: number;
    has_legacy_credential: boolean;
  }>;
  entries: ApiPoolEntry[];
  usage: {
    totals: { today: ApiPoolUsageTotals; month: ApiPoolUsageTotals };
    daily: Array<{ day: string; requests: number; failures: number; tokens_in: number; tokens_out: number }>;
  };
  queue: {
    pending: number;
    in_flight: number;
    done: number;
    failed: number;
    total: number;
    oldest_failed_at: number | null;
    last_24h: { total: number; ok: number };
    failed_items: ApiPoolQueueItem[];
  };
  cache: { entries: number; hits: number; newest_at: number | null };
  plan: Record<string, string[]>;
}

/** One provider's credential state, as the translation-engine picker needs it. */
export interface ProviderKeyStatus {
  has_key: boolean;
  /** How many API-pool configurations exist for this provider. */
  pool_entries?: number;
  /** How many can serve right now (enabled, not parked, within their limits). */
  pool_usable?: number;
}
