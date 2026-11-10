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

/** One provider's credential state, as the translation-engine picker needs it. */
export interface ProviderKeyStatus {
  has_key: boolean;
}

/**
 * One Chrome profile on this PC, exactly as `/api/chrome/profiles` reports it.
 *
 * `emails` comes from Chrome's own metadata (Local State + Preferences); the
 * panel masks it before rendering. Nothing here is a secret.
 */
export interface ChromeProfile {
  id: string;
  root_path: string;
  /** 'stable' | 'beta' | 'canary' | 'custom' */
  root_channel: string;
  profile_folder: string;
  display_name: string;
  emails: string[];
  path: string;
  has_google_account: boolean;
  is_locked: boolean;
  /** 'in_use' when Chrome has this profile open; 'chrome_running' when that cannot be told. */
  locked_reason: '' | 'in_use' | 'chrome_running';
  is_last_used: boolean;
  warnings: string[];
}

export type ChromeProfileStatus = 'ready' | 'login_required' | 'locked' | 'error';

/** One `/api/chrome/profiles/validate` answer. */
export interface ChromeProfileValidation {
  success: boolean;
  profile_id: string;
  status: ChromeProfileStatus | 'manual_login';
  message: string;
  url?: string;
  /** 'working_copy' when Chrome's remote-debugging restriction forced a private copy. */
  via?: 'direct' | 'working_copy';
}

export interface ChromeProfilesSnapshot {
  profiles: ChromeProfile[];
  selected: ChromeProfile | null;
  selected_profile_id: string;
  selected_missing: boolean;
  last_selected_profile_folder: string;
  last_selected_root_path: string;
  headless: boolean;
  roots: string[];
  chrome_path: string;
  playwright_ready: boolean;
  chrome_running: boolean;
  errors: string[];
}
