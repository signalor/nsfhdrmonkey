/**
 * Script to package submission files into a zip archive.
 *
 * Creates a submission.zip containing:
 * - model.py (the submission model)
 * - model_affi.pth (model weights for affi)
 * - model_beignet.pth (model weights for beignet)
 * - train_data_average_std_affi.npz (normalization stats for affi)
 * - train_data_average_std_beignet.npz (normalization stats for beignet)
 *
 * Usage: npx ts-node zip.ts
 *    or: bun run zip.ts
 */

import * as fs from "fs";
import * as path from "path";
import { execSync } from "child_process";

const SUBMISSION_DIR = "./submission";
const CHECKPOINTS_DIR = "./checkpoints";
const OUTPUT_ZIP = "./submission.zip";

interface FileMapping {
  source: string;
  dest: string;
}

async function main() {
  console.log("Preparing submission package...\n");

  // Files to include in the submission
  const files: FileMapping[] = [
    // Model code
    {
      source: path.join(SUBMISSION_DIR, "model.py"),
      dest: "model.py",
    },
    // Model weights
    {
      source: path.join(CHECKPOINTS_DIR, "best_model_affi.pt"),
      dest: "model_affi.pth",
    },
    {
      source: path.join(CHECKPOINTS_DIR, "best_model_beignet.pt"),
      dest: "model_beignet.pth",
    },
    // Normalization stats
    {
      source: path.join(CHECKPOINTS_DIR, "norm_stats_affi.npz"),
      dest: "train_data_average_std_affi.npz",
    },
    {
      source: path.join(CHECKPOINTS_DIR, "norm_stats_beignet.npz"),
      dest: "train_data_average_std_beignet.npz",
    },
  ];

  // Check that all source files exist
  console.log("Checking source files...");
  const missingFiles: string[] = [];
  for (const file of files) {
    if (!fs.existsSync(file.source)) {
      missingFiles.push(file.source);
      console.log(`  ❌ Missing: ${file.source}`);
    } else {
      const stats = fs.statSync(file.source);
      const sizeMB = (stats.size / (1024 * 1024)).toFixed(2);
      console.log(`  ✓ Found: ${file.source} (${sizeMB} MB)`);
    }
  }

  if (missingFiles.length > 0) {
    console.error("\n❌ Error: Some required files are missing.");
    console.error(
      "Please ensure you have trained the model and have all checkpoint files.",
    );
    process.exit(1);
  }

  // Create a temporary directory for staging
  const tempDir = "./submission_temp";
  if (fs.existsSync(tempDir)) {
    fs.rmSync(tempDir, { recursive: true });
  }
  fs.mkdirSync(tempDir, { recursive: true });

  // Copy files to temp directory with correct names
  console.log("\nStaging files...");
  for (const file of files) {
    const destPath = path.join(tempDir, file.dest);
    fs.copyFileSync(file.source, destPath);
    console.log(`  Copied: ${file.source} -> ${file.dest}`);
  }

  // Convert checkpoint files to proper format if needed
  // The checkpoints contain full state dicts with optimizer, we need just model weights
  console.log("\nExtracting model weights from checkpoints...");

  // Create a Python script to extract just the model state dict
  const extractScript = `
import torch
import sys

def extract_weights(input_path, output_path):
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    torch.save(state_dict, output_path)
    print(f"  Extracted weights: {output_path}")

extract_weights('${tempDir}/model_affi.pth', '${tempDir}/model_affi.pth')
extract_weights('${tempDir}/model_beignet.pth', '${tempDir}/model_beignet.pth')
`;

  fs.writeFileSync("./extract_weights.py", extractScript);

  try {
    execSync("python extract_weights.py", { stdio: "inherit" });
  } catch (e) {
    console.error("Failed to extract weights. Trying with python3...");
    try {
      execSync("python3 extract_weights.py", { stdio: "inherit" });
    } catch (e2) {
      console.error(
        "Failed to extract weights. Please ensure Python and PyTorch are installed.",
      );
      process.exit(1);
    }
  }

  fs.unlinkSync("./extract_weights.py");

  // Copy normalization stats (keep original format with mean/std, shape (1, 1, C, F))
  console.log("\nCopying normalization stats...");

  const convertStatsScript = `
import numpy as np
import sys

def copy_stats(input_path, output_path):
    data = np.load(input_path)
    # Keep the original format: mean/std with shape (1, 1, C, F)
    mean = data['mean']
    std = data['std']
    np.savez(output_path, mean=mean, std=std)
    print(f"  Copied stats: {output_path} (mean shape: {mean.shape})")

copy_stats('${tempDir}/train_data_average_std_affi.npz', '${tempDir}/train_data_average_std_affi.npz')
copy_stats('${tempDir}/train_data_average_std_beignet.npz', '${tempDir}/train_data_average_std_beignet.npz')
`;

  fs.writeFileSync("./convert_stats.py", convertStatsScript);

  try {
    execSync("python convert_stats.py", { stdio: "inherit" });
  } catch (e) {
    try {
      execSync("python3 convert_stats.py", { stdio: "inherit" });
    } catch (e2) {
      console.error(
        "Failed to convert stats. Please ensure Python and NumPy are installed.",
      );
      process.exit(1);
    }
  }

  fs.unlinkSync("./convert_stats.py");

  // Create zip file
  console.log("\nCreating zip archive...");

  // Remove existing zip if present
  if (fs.existsSync(OUTPUT_ZIP)) {
    fs.unlinkSync(OUTPUT_ZIP);
  }

  // Use platform-appropriate zip command
  const isWindows = process.platform === "win32";

  try {
    if (isWindows) {
      // Use PowerShell on Windows
      execSync(
        `powershell -Command "Compress-Archive -Path '${tempDir}/*' -DestinationPath '${OUTPUT_ZIP}'"`,
        { stdio: "inherit" },
      );
    } else {
      // Use zip on Unix-like systems
      execSync(`cd ${tempDir} && zip -r ../submission.zip ./*`, {
        stdio: "inherit",
      });
    }
  } catch (e) {
    console.error(
      "Failed to create zip. Please ensure zip utility is available.",
    );
    process.exit(1);
  }

  // Clean up temp directory
  fs.rmSync(tempDir, { recursive: true });

  // Report final zip size
  const zipStats = fs.statSync(OUTPUT_ZIP);
  const zipSizeMB = (zipStats.size / (1024 * 1024)).toFixed(2);

  console.log("\n" + "=".repeat(50));
  console.log(`✓ Submission package created: ${OUTPUT_ZIP}`);
  console.log(`  Size: ${zipSizeMB} MB`);
  console.log("=".repeat(50));

  // List contents of zip
  console.log("\nPackage contents:");
  files.forEach((f) => console.log(`  - ${f.dest}`));
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
