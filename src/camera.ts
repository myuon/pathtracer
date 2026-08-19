export type Vec3 = [number, number, number];

const EPS = 1e-4;

function normalize(v: Vec3): Vec3 {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

export interface CameraBasis {
  position: Vec3;
  /** 右方向 */
  u: Vec3;
  /** 上方向 */
  v: Vec3;
  /** 前方向 (シーンに向かう) */
  w: Vec3;
  tanHalfFov: number;
  focusDist: number;
  lensRadius: number;
}

/** target を中心に回る orbit カメラ */
export class OrbitCamera {
  target: Vec3 = [0, 1, 0];
  distance = 12;
  /** 水平角 (rad) */
  yaw = Math.PI * 0.5;
  /** 仰角 (rad) */
  pitch = 0.14;
  /** 垂直画角 (deg) */
  fovDeg = 32;
  /** レンズ半径。0 で被写界深度なし */
  aperture = 0.08;
  /** true なら焦点距離 = target までの距離 */
  autoFocus = true;
  manualFocusDist = 10;

  dirty = true;

  position(): Vec3 {
    const cp = Math.cos(this.pitch);
    return [
      this.target[0] + this.distance * cp * Math.cos(this.yaw),
      this.target[1] + this.distance * Math.sin(this.pitch),
      this.target[2] + this.distance * cp * Math.sin(this.yaw),
    ];
  }

  basis(): CameraBasis {
    const position = this.position();
    const w = normalize([
      this.target[0] - position[0],
      this.target[1] - position[1],
      this.target[2] - position[2],
    ]);
    const u = normalize(cross(w, [0, 1, 0]));
    const v = cross(u, w);
    return {
      position,
      u,
      v,
      w,
      tanHalfFov: Math.tan((this.fovDeg * Math.PI) / 360),
      focusDist: this.autoFocus ? this.distance : this.manualFocusDist,
      lensRadius: this.aperture * 0.5,
    };
  }

  orbit(dx: number, dy: number) {
    this.yaw += dx;
    this.pitch = Math.max(-Math.PI / 2 + EPS, Math.min(Math.PI / 2 - EPS, this.pitch + dy));
    this.dirty = true;
  }

  dolly(factor: number) {
    this.distance = Math.max(0.5, Math.min(200, this.distance * factor));
    this.dirty = true;
  }

  pan(dx: number, dy: number) {
    const { u, v } = this.basis();
    const scale = this.distance * 0.0015;
    for (let i = 0; i < 3; i++) {
      this.target[i] += (-u[i] * dx + v[i] * dy) * scale;
    }
    this.dirty = true;
  }
}

export interface CameraInputCallbacks {
  /** ドラッグ・ホイール中かどうかが変わったとき */
  onInteractingChange?: (interacting: boolean) => void;
}

/** canvas にマウス/タッチ操作を結びつける */
export function attachCameraControls(
  canvas: HTMLCanvasElement,
  camera: OrbitCamera,
  cb: CameraInputCallbacks = {},
): void {
  let dragButton = -1;
  let lastX = 0;
  let lastY = 0;
  let interacting = false;
  let wheelTimer = 0;

  const setInteracting = (next: boolean) => {
    if (interacting === next) return;
    interacting = next;
    cb.onInteractingChange?.(next);
  };

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  canvas.addEventListener("pointerdown", (e) => {
    if (dragButton !== -1) return;
    dragButton = e.button;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
    setInteracting(true);
  });

  canvas.addEventListener("pointermove", (e) => {
    if (dragButton === -1) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    if (dragButton === 0) {
      camera.orbit(dx * 0.006, dy * 0.006);
    } else {
      camera.pan(dx, dy);
    }
  });

  const endDrag = (e: PointerEvent) => {
    if (dragButton === -1) return;
    dragButton = -1;
    canvas.releasePointerCapture?.(e.pointerId);
    setInteracting(false);
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);

  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      camera.dolly(Math.exp(e.deltaY * 0.001));
      setInteracting(true);
      clearTimeout(wheelTimer);
      wheelTimer = window.setTimeout(() => {
        if (dragButton === -1) setInteracting(false);
      }, 120);
    },
    { passive: false },
  );
}
