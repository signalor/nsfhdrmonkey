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
    // Model weights (from checkpoints)
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
      console.log(`  Missing: ${file.source}`);
    } else {
      const stats = fs.statSync(file.source);
      const sizeMB = (stats.size / (1024 * 1024)).toFixed(2);
      console.log(`  Found: ${file.source} (${sizeMB} MB)`);
    }
  }

  if (missingFiles.length > 0) {
    console.error("\nError: Some required files are missing.");
    console.error(
      "Please ensure you have trained the model and have all checkpoint files."
    );
    process.exit(1);
  }

  // Create temporary staging directory
  const tempDir = "./submission_temp";
  if (fs.existsSync(tempDir)) {
    fs.rmSync(tempDir, { recursive: true });
  }
  fs.mkdirSync(tempDir, { recursive: true });

  // Copy files with correct names
  console.log("\nStaging files...");
  for (const file of files) {
    const destPath = path.join(tempDir, file.dest);
    fs.copyFileSync(file.source, destPath);
    console.log(`  Copied: ${file.source} -> ${file.dest}`);
  }

  // Extract model weights from checkpoints
  console.log("\nExtracting model weights...");

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
    print(f"  Extracted: {output_path}")

extract_weights('${tempDir}/model_affi.pth', '${tempDir}/model_affi.pth')
extract_weights('${tempDir}/model_beignet.pth', '${tempDir}/model_beignet.pth')
`;

  fs.writeFileSync("./extract_weights.py", extractScript);

  try {
    execSync("python extract_weights.py", { stdio: "inherit" });
  } catch (e) {
    try {
      execSync("python3 extract_weights.py", { stdio: "inherit" });
    } catch (e2) {
      console.error("Failed to extract weights.");
      process.exit(1);
    }
  }

  fs.unlinkSync("./extract_weights.py");

  // Remove existing zip
  if (fs.existsSync(OUTPUT_ZIP)) {
    fs.unlinkSync(OUTPUT_ZIP);
  }

  // Create zip
  console.log("\nCreating zip archive...");
  const isWindows = process.platform === "win32";

  try {
    if (isWindows) {
      execSync(
        `powershell -Command "Compress-Archive -Path '${tempDir}/*' -DestinationPath '${OUTPUT_ZIP}'"`,
        { stdio: "inherit" }
      );
    } else {
      execSync(`cd ${tempDir} && zip -r ../submission.zip ./*`, {
        stdio: "inherit",
      });
    }
  } catch (e) {
    console.error("Failed to create zip.");
    process.exit(1);
  }

  // Cleanup
  fs.rmSync(tempDir, { recursive: true });

  // Report
  const zipStats = fs.statSync(OUTPUT_ZIP);
  const zipSizeMB = (zipStats.size / (1024 * 1024)).toFixed(2);

  console.log("\n" + "=".repeat(50));
  console.log(`Submission package created: ${OUTPUT_ZIP}`);
  console.log(`Size: ${zipSizeMB} MB`);
  console.log("=".repeat(50));

  console.log("\nContents:");
  files.forEach((f) => console.log(`  - ${f.dest}`));
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
