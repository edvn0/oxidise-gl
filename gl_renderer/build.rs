use std::process::Command;

fn main() {
    validate("src/shader.vert");
    validate("src/shader.frag");
    validate("src/cull.comp");
    println!("cargo:rerun-if-changed=src/shader.vert");
    println!("cargo:rerun-if-changed=src/shader.frag");
    println!("cargo:rerun-if-changed=src/cull.comp");
}

fn validate(path: &str) {
    let out = Command::new("glslangValidator")
        .arg(path)
        .output()
        .unwrap_or_else(|_| {
            panic!(
                "glslangValidator not found — install the Vulkan SDK and ensure it is on PATH\n\
                 (https://vulkan.lunarg.com/sdk/home)"
            )
        });

    if !out.status.success() {
        let log = String::from_utf8_lossy(&out.stdout);
        panic!("Shader validation failed for {path}:\n{log}");
    }
}
