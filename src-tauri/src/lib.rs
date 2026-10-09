use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;
use tauri::{Manager, WindowEvent};

struct LocalBackend {
    child: Mutex<Option<Child>>,
}

impl Drop for LocalBackend {
    fn drop(&mut self) {
        if let Ok(Some(child)) = self.child.get_mut() {
            let _ = child.kill();
        }
    }
}

fn backend_is_running() -> bool {
    let address = SocketAddr::from(([127, 0, 0, 1], 8001));
    TcpStream::connect_timeout(&address, Duration::from_millis(200)).is_ok()
}

fn spawn_backend(mut command: Command) -> std::io::Result<Child> {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }

    command.spawn()
}

fn sidecar_candidates(app: &tauri::AppHandle) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(resource_dir) = app.path().resource_dir() {
        candidates.push(resource_dir.join("binaries").join("dubvibe-backend-complete").join("dubvibe-backend-complete.exe"));
    }
    if let Ok(executable) = std::env::current_exe() {
        if let Some(executable_dir) = executable.parent() {
            candidates.push(
                executable_dir
                    .join("binaries")
                    .join("dubvibe-backend-complete")
                    .join("dubvibe-backend-complete.exe"),
            );
            // Legacy single-file sidecar layout, kept as a fallback.
            candidates.push(executable_dir.join("binaries").join("dubvibe-backend-complete.exe"));
        }
    }
    candidates
}

fn find_project_root() -> Option<PathBuf> {
    // Walk up from the current directory until a directory containing
    // backend/app/main.py is found (the repo root).
    let mut dir = std::env::current_dir().ok()?;
    loop {
        if dir.join("backend").join("app").join("main.py").is_file() {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

fn start_backend(app: &tauri::AppHandle) -> Option<Child> {
    if backend_is_running() {
        return None;
    }

    if let Some(sidecar) = sidecar_candidates(app)
        .into_iter()
        .find(|candidate| candidate.is_file())
    {
        return spawn_backend(Command::new(sidecar)).ok();
    }

    if cfg!(debug_assertions) {
        let python_cmd = std::env::var("PYTHON").unwrap_or_else(|_| "python".to_string());
        let mut command = Command::new(python_cmd);
        // uvicorn must run from the repo root for `backend.app.main` to be
        // importable; the dev binary's cwd is src-tauri under `tauri dev`.
        if let Some(project_root) = find_project_root() {
            command.current_dir(project_root);
        }
        command.args([
            "-m",
            "uvicorn",
            "backend.app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8001",
        ]);
        return spawn_backend(command).ok();
    }

    None
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            app.manage(LocalBackend {
                child: Mutex::new(start_backend(app.handle())),
            });
            let window = app.get_webview_window("main").unwrap();
            let app_handle = app.handle().clone();
            window.on_window_event(move |event| {
                if let WindowEvent::CloseRequested { .. } = event {
                    if let Some(backend) = app_handle.try_state::<LocalBackend>() {
                        if let Ok(mut guard) = backend.child.lock() {
                            if let Some(mut child) = guard.take() {
                                let _ = child.kill();
                            }
                        }
                    }
                    app_handle.exit(0);
                }
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
