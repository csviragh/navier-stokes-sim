/**
 * Mouse / touch / pen input via Pointer Events. Each active pointer's motion
 * since the previous frame is collected as a list of segments in normalised
 * canvas coordinates (x right, y up, both in [0, 1]) with a velocity in
 * normalised units per second.
 */
export interface Stroke {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  /** Pointer velocity, normalised units per second. */
  vx: number;
  vy: number;
}

interface Tracked {
  x: number;
  y: number;
  t: number;
  vx: number;
  vy: number;
}

export class PointerInput {
  private readonly active = new Map<number, Tracked>();
  private strokes: Stroke[] = [];

  constructor(private readonly el: HTMLElement) {
    el.addEventListener('pointerdown', this.down);
    el.addEventListener('pointermove', this.move);
    el.addEventListener('pointerup', this.up);
    el.addEventListener('pointercancel', this.up);
    el.addEventListener('lostpointercapture', this.up);
    el.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  private norm(e: PointerEvent): [number, number] {
    const r = this.el.getBoundingClientRect();
    return [(e.clientX - r.left) / r.width, 1 - (e.clientY - r.top) / r.height];
  }

  private down = (e: PointerEvent): void => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    this.el.setPointerCapture(e.pointerId);
    const [x, y] = this.norm(e);
    this.active.set(e.pointerId, { x, y, t: e.timeStamp, vx: 0, vy: 0 });
    // a click without motion still drops some dye
    this.strokes.push({ x0: x, y0: y, x1: x, y1: y, vx: 0, vy: 0 });
  };

  private move = (e: PointerEvent): void => {
    const p = this.active.get(e.pointerId);
    if (!p) return;
    const events = typeof e.getCoalescedEvents === 'function' ? e.getCoalescedEvents() : [];
    for (const ev of events.length ? events : [e]) {
      const [x, y] = this.norm(ev);
      const dt = Math.max((ev.timeStamp - p.t) / 1000, 1 / 240);
      // smooth the velocity estimate a little
      p.vx = 0.4 * p.vx + 0.6 * ((x - p.x) / dt);
      p.vy = 0.4 * p.vy + 0.6 * ((y - p.y) / dt);
      this.strokes.push({ x0: p.x, y0: p.y, x1: x, y1: y, vx: p.vx, vy: p.vy });
      p.x = x;
      p.y = y;
      p.t = ev.timeStamp;
    }
  };

  private up = (e: PointerEvent): void => {
    this.active.delete(e.pointerId);
  };

  /** Return and clear the strokes collected since the last call. */
  take(): Stroke[] {
    const s = this.strokes;
    this.strokes = [];
    return s;
  }
}
