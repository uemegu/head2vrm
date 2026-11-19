import * as THREE from 'three';

// The closed lids' skin without the lash tips painted on the texture, blended in by how far each eye is closed.
// Long lashes painted on the skin texture; closing the eye stretches the lid skin over the eye, and the lashes' tips there read
// as stray lines. The open eye keeps them. The patch sits beside the model: NAME.vrm -> NAME-closedlid.json and .png
// (make_face.sh writes them with tools/tripo/pipeline/closed_lid_patch.py: the cleaned lid, alpha = the band, and the
// patch's rect in the model's UVs, one rect per eye). Other apps do not use it; the VRM works without it.
const ON = [0.2, 0.85];   // the eye's closure (0 open, 1 shut) over which the cleaned lid fades in

export async function createClosedLid(vrm, url) {
  const metaUrl = url.split(/[?#]/)[0].replace(/\.vrm$/i, '-closedlid.json');
  const res = await fetch(metaUrl, { cache: 'no-cache' });
  if (!res.ok) return null;
  const meta = await res.json();   // its image carries ?v=<when it was made>
  const tex = await new THREE.TextureLoader().loadAsync(new URL(meta.image, new URL(metaUrl, location.href)).href);
  tex.colorSpace = THREE.SRGBColorSpace; tex.flipY = false; tex.generateMipmaps = false;
  tex.minFilter = tex.magFilter = THREE.LinearFilter;
  const eyes = meta.eyes.slice(0, 2);
  const uniforms = {
    uClPatch: { value: tex },
    uClRect: { value: new THREE.Vector4(...meta.rect) },
    uClEyeA: { value: new THREE.Vector4(...eyes[0].rect) },
    uClEyeB: { value: new THREE.Vector4(...(eyes[1]?.rect ?? [2, 2, 2, 2])) },
    uClA: { value: 0 }, uClB: { value: 0 },
    uClDebug: { value: 0 },   // 1: the patch drawn green (its coverage)
  };
  // each eye's closure from the meshes' own morph targets: the lid's vertices (UVs in the eye's rect), how far each
  // target moves them, against the target that moves them most (the blink)
  const sources = [];
  vrm.scene.traverse((o) => {
    if (!o.isMesh || !o.morphTargetDictionary || !o.geometry.attributes.uv || !o.geometry.morphAttributes.position) return;
    if (!Object.keys(o.morphTargetDictionary).some((n) => /blink/i.test(n))) return;   // the body with the lids, not the eyeballs
    const uv = o.geometry.attributes.uv, targets = o.geometry.morphAttributes.position;
    const perEye = eyes.map((e) => {
      const ids = [];
      for (let i = 0; i < uv.count; i++) {
        const u = uv.getX(i), v = uv.getY(i);
        if (u > e.rect[0] && u < e.rect[2] && v > e.rect[1] && v < e.rect[3]) ids.push(i);
      }
      if (!ids.length) return null;
      const disp = targets.map((t) => ids.reduce((s, i) => s + Math.hypot(t.getX(i), t.getY(i), t.getZ(i)), 0) / ids.length);
      const ref = Math.max(...disp);
      return ref > 1e-5 ? disp.map((d) => d / ref) : null;
    });
    if (perEye.some(Boolean)) sources.push({ mesh: o, perEye });
  });
  if (!sources.length) return null;
  const KEY = 'closed-lid-v2';
  const patch = (m) => {
    if (!m || m.userData.closedLid || !m.map) return;
    m.userData.closedLid = true;
    const before = m.onBeforeCompile, key = m.customProgramCacheKey();
    m.onBeforeCompile = (shader, renderer) => {
      before?.call(m, shader, renderer);
      Object.assign(shader.uniforms, uniforms);
      // clSample: the map with the cleaned lid blended in. Used right after the map is read, and in place of every other
      // read of the map at vMapUv
      shader.fragmentShader = shader.fragmentShader
        .replace(/texture2D\(\s*map\s*,\s*vMapUv\s*([,)])/g, 'clSample( vMapUv$1')
        .replace('#include <map_pars_fragment>', `#include <map_pars_fragment>
uniform sampler2D uClPatch;
uniform vec4 uClRect, uClEyeA, uClEyeB;
uniform float uClA, uClB, uClDebug;
#ifdef USE_MAP
float clWeight(vec2 uv) {
  return (all(greaterThan(uv, uClEyeA.xy)) && all(lessThan(uv, uClEyeA.zw))) ? uClA
       : (all(greaterThan(uv, uClEyeB.xy)) && all(lessThan(uv, uClEyeB.zw))) ? uClB : 0.0;
}
vec4 clBlend(vec4 c, vec2 uv) {
  float cw = clWeight(uv);
  if (cw <= 0.0) return c;
  vec4 pc = texture2D(uClPatch, (uv - uClRect.xy) / (uClRect.zw - uClRect.xy));
  return vec4(mix(c.rgb, mix(pc.rgb, vec3(0.0, 1.0, 0.0), uClDebug), pc.a * cw), c.a);
}
vec4 clSample(vec2 uv) { return clBlend(texture2D(map, uv), uv); }
vec4 clSample(vec2 uv, float bias) { return clBlend(texture2D(map, uv, bias), uv); }
#endif`)
        .replace('#include <map_fragment>', `#include <map_fragment>
#ifdef USE_MAP
      if (clWeight(vMapUv) > 0.0) diffuseColor.rgb = clBlend(vec4(diffuseColor.rgb / max(diffuse, vec3(1e-4)), 1.0), vMapUv).rgb * diffuse;
#endif`);
    };
    m.customProgramCacheKey = () => `${key}-${KEY}`;
    m.needsUpdate = true;
  };
  const closure = (k) => {
    let c = 0;
    for (const s of sources) {
      const w = s.perEye[k], inf = s.mesh.morphTargetInfluences;
      if (!w || !inf) continue;
      let sum = 0;
      for (let i = 0; i < w.length; i++) sum += (inf[i] || 0) * w[i];
      c = Math.max(c, sum);
    }
    return THREE.MathUtils.smoothstep(Math.min(c, 1), ON[0], ON[1]);
  };
  return {
    enabled: true,
    // every frame, after the expressions: patch the current materials (a viewer may swap them)
    update() {
      for (const s of sources) [].concat(s.mesh.material).forEach(patch);
      uniforms.uClA.value = this.enabled ? closure(0) : 0;
      uniforms.uClB.value = this.enabled ? closure(1) : 0;
    },
    set debug(v) { uniforms.uClDebug.value = v ? 1 : 0; },
    get weights() { return [uniforms.uClA.value, uniforms.uClB.value]; },
  };
}
