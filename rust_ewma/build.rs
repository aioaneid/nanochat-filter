#[cfg(all(feature = "metal", feature = "python"))]
use cc;

fn main() {
    let features: Vec<String> = std::env::vars()
        .filter(|(k, _)| k.starts_with("CARGO_FEATURE_"))
        .map(|(k, _)| k)
        .collect();
    println!("cargo:warning=ACTIVE_FEATURES={:?}", features);

    #[cfg(all(feature = "metal", feature = "python"))]
    {
        let py_exec = std::env::var("PYTHON_SYS_EXECUTABLE")
            .or_else(|_| std::env::var("PYO3_PYTHON"))
            .unwrap_or_else(|_| "python3".to_string());
        println!("cargo:warning=BUILD_RS_PY_EXEC_FINAL={}", py_exec);

        // 1. Get PyTorch include paths
        let torch_path_cmd = std::process::Command::new(&py_exec)
            .args(["-c", "import torch; from torch.utils import cpp_extension; print(';'.join(cpp_extension.include_paths()))"])
            .output();

        let torch_includes: Vec<String> = match torch_path_cmd {
            Ok(output) if output.status.success() => {
                let torch_raw = String::from_utf8_lossy(&output.stdout);
                torch_raw
                    .trim()
                    .split(';')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect()
            }
            _ => {
                println!("cargo:warning=TORCH_QUERY_FAILED: trying common paths");
                // Falling back to standard Mac locations if torch is installed via standard venv/conda
                vec![] // Let's keep it empty for now and see if we can find more
            }
        };
        println!("cargo:warning=TORCH_INCLUDES={:?}", torch_includes);

        // 2. Get Python include paths
        let py_path_cmd = std::process::Command::new(&py_exec)
            .args(["-c", "import sysconfig; print(f\"{sysconfig.get_path('include')};{sysconfig.get_config_var('CONFINCLUDEPY')}\")"])
            .output();

        let py_includes: Vec<String> = match py_path_cmd {
            Ok(output) if output.status.success() => {
                let py_raw = String::from_utf8_lossy(&output.stdout);
                py_raw
                    .trim()
                    .split(';')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect()
            }
            _ => vec![],
        };
        println!("cargo:warning=PY_INCLUDES={:?}", py_includes);

        // 3. Build the bridge
        // let mut build = cc::Build::new();
        // build
        //     .file("src/mps_bridge.mm")
        //     .flag("-std=c++17")
        //     .flag("-fobjc-arc")
        //     .cpp(true);

        let mut builder = cc::Build::new();
        builder
            .cpp(true)
            .std("c++17")
            .file("src/mps_bridge.mm")
            .flag("-Wall")
            .flag("-Wextra");

        // Apply -isystem to each torch include path individually
        for path in torch_includes.iter().chain(py_includes.iter()) {
            builder.flag(&format!("-isystem{}", path));
        }

        builder.compile("mps_bridge");

        // for path in torch_includes {
        //     build.include(path);
        // }
        // for path in py_includes {
        //     build.include(path);
        // }

        // build.compile("mps_bridge");

        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rerun-if-changed=src/mps_bridge.mm");
    }
}
