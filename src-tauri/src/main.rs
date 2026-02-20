// Prevents additional console window on Windows in release.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;
use tauri::{Emitter, Manager, State};
use tauri_plugin_shell::ShellExt;

/// Stores the backend server port once discovered.
struct BackendState {
    port: Mutex<Option<u16>>,
}

/// Tauri command: returns the backend URL for the frontend to use.
#[tauri::command]
fn get_backend_url(state: State<BackendState>) -> String {
    let port = state.port.lock().unwrap();
    match *port {
        Some(p) => format!("http://127.0.0.1:{}", p),
        None => String::new(),
    }
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(BackendState {
            port: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![get_backend_url])
        .setup(|app| {
            let app_handle = app.handle().clone();
            let state = app_handle.state::<BackendState>().inner().clone();

            // Spawn the Python sidecar
            let shell = app_handle.shell();
            let sidecar = shell
                .sidecar("binaries/knwler-server")
                .expect("failed to create sidecar command");

            let (mut rx, _child) = sidecar.spawn().expect("failed to spawn sidecar");

            // Read stdout to discover the port
            tauri::async_runtime::spawn(async move {
                use tauri_plugin_shell::process::CommandEvent;

                while let Some(event) = rx.recv().await {
                    match event {
                        CommandEvent::Stdout(line) => {
                            let line_str = String::from_utf8_lossy(&line);
                            for segment in line_str.lines() {
                                if let Some(port_str) = segment.strip_prefix("PORT:") {
                                    if let Ok(port) = port_str.trim().parse::<u16>() {
                                        {
                                            let mut p = state.port.lock().unwrap();
                                            *p = Some(port);
                                        }
                                        // Tell the frontend the backend is ready
                                        let _ = app_handle.emit("backend-ready", port);
                                        eprintln!("Sidecar ready on port {}", port);
                                    }
                                }
                            }
                        }
                        CommandEvent::Stderr(line) => {
                            eprintln!(
                                "sidecar: {}",
                                String::from_utf8_lossy(&line)
                            );
                        }
                        _ => {}
                    }
                }
            });

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
