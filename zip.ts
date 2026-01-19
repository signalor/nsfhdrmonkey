/**
 * Script to package submission files into a zip archive.
 * Usage: npx ts-node zip.ts
 */

import * as fs from "fs";
import * as path from "path";
import { execSync } from "child_process";

// Helper to parse command line arguments
function getArg(argName: string, defaultValue: string): string {
  const arg = process.argv.find((a) => a.startsWith(`--${argName}=`));
  if (arg) {
    return arg.split("=")[1];
  }
  return defaultValue;
}

// Configuration
// UPDATED: Defaults to "." (current dir) for model.py and "checkpoints" for weights
const SUBMISSION_CODE_DIR = getArg("submission_dir", "./submission"); 
const CHECKPOINTS_DIR = getArg("checkpoints_dir", "./ncheckpoints");
const OUTPUT_ZIP = getArg("output_zip", "./submission.zip");

interface FileMapping {
  source: string;
  dest: string;
}

async function main() {
  console.log("Preparing submission package...\n");

  const files: FileMapping[] = [
    // 1. The Model Code
    {
      source: path.join(SUBMISSION_CODE_DIR, "model.py"),
      dest: "model.py",
    },
    // 2. Monkey A (Affi)
    {
      source: path.join(CHECKPOINTS_DIR, "model_affi.pth"),
      dest: "model_affi.pth",
    },
    {
      source: path.join(CHECKPOINTS_DIR, "train_data_average_std_affi.npz"),
      dest: "train_data_average_std_affi.npz",
    },
    // 3. Monkey B (Beignet)
    {
      source: path.join(CHECKPOINTS_DIR, "model_beignet.pth"),
      dest: "model_beignet.pth",
    },
    {
      source: path.join(CHECKPOINTS_DIR, "train_data_average_std_beignet.npz"),
      dest: "train_data_average_std_beignet.npz",
    },
  ];

  // Temporary directory for zipping
  const tempDir = "./temp_submission";
  if (fs.existsSync(tempDir)) {
    fs.rmSync(tempDir, { recursive: true });
  }
  fs.mkdirSync(tempDir);

  // Copy files
  console.log("Copying files:");
  for (const file of files) {
    if (!fs.existsSync(file.source)) {
      console.error(`❌ Missing expected file: ${file.source}`);
      console.error(`   Please run ntrainer.py first to generate checkpoints.`);
      process.exit(1);
    }
    
    fs.copyFileSync(file.source, path.join(tempDir, file.dest));
    console.log(`  ✅ ${file.dest}`);
  }

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
    console.log(`\n🎉 Submission ready: ${OUTPUT_ZIP}`);
  } catch (e) {
    console.error("Failed to create zip.");
    process.exit(1);
  }

  // Cleanup
  fs.rmSync(tempDir, { recursive: true });
}

main();