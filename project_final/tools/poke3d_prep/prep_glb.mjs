#!/usr/bin/env node
// Decode a Pokemon-3D-api optimized GLB into something Assimp 5.2 can load:
// strip Draco + WebP, drop skins/joint weights, rewrite textures as PNG.
//
// Usage: node prep_glb.mjs <input.glb> <output.glb>

import { NodeIO } from "@gltf-transform/core";
import { ALL_EXTENSIONS } from "@gltf-transform/extensions";
import { dedup, flatten, textureCompress, weld } from "@gltf-transform/functions";
import draco3d from "draco3dgltf";
import sharp from "sharp";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";

async function main() {
  const input = process.argv[2];
  const output = process.argv[3];
  if (!input || !output) {
    console.error("Usage: node prep_glb.mjs <input.glb> <output.glb>");
    process.exit(2);
  }

  const io = new NodeIO()
    .registerExtensions(ALL_EXTENSIONS)
    .registerDependencies({
      "draco3d.decoder": await draco3d.createDecoderModule(),
    });

  const doc = await io.read(input);

  for (const skin of [...doc.getRoot().listSkins()]) {
    skin.dispose();
  }
  for (const anim of [...doc.getRoot().listAnimations()]) {
    anim.dispose();
  }
  for (const mesh of doc.getRoot().listMeshes()) {
    for (const prim of mesh.listPrimitives()) {
      prim.setAttribute("JOINTS_0", null);
      prim.setAttribute("WEIGHTS_0", null);
    }
  }

  await doc.transform(
    flatten(),
    weld(),
    dedup(),
    textureCompress({ encoder: sharp, targetFormat: "png" }),
  );

  // Drop leftover extension declarations so writers do not demand an encoder.
  for (const ext of [...doc.getRoot().listExtensionsUsed()]) {
    if (
      ext.extensionName === "KHR_draco_mesh_compression" ||
      ext.extensionName === "EXT_texture_webp"
    ) {
      ext.dispose();
    }
  }

  mkdirSync(dirname(output), { recursive: true });
  await io.write(output, doc);
  console.log(`prep_glb: ${input} -> ${output}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
