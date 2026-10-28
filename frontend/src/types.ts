export type ProjectMode = 'EDIT' | 'DUB';
export type ProjectStatus = 'Ready' | 'Processing' | 'Queued' | 'Exporting';
export type ProviderMode = 'LOCAL AI' | 'FREE ONLINE' | 'PAID API';

export interface ProviderStatus {
  id: string;
  name: string;
  mode: ProviderMode;
  connected: boolean;
  primary: boolean;
  fallback: boolean;
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
