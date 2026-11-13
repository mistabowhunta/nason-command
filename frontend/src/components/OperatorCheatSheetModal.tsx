import { useEffect } from 'react';
import {
  AlertTriangle,
  BookOpen,
  Check,
  Cpu,
  Eye,
  ShieldCheck,
  Upload,
  UserCheck,
  X,
} from 'lucide-react';

interface OperatorCheatSheetModalProps {
  open: boolean;
  onClose: () => void;
}

export function OperatorCheatSheetModal({ open, onClose }: OperatorCheatSheetModalProps) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      onClick={onClose}
      className="fixed inset-0 z-50 bg-black/85 backdrop-blur-xs flex items-center justify-center p-2 sm:p-6 overflow-y-auto"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="bg-zinc-950 border border-emerald-500/40 rounded-xl max-w-4xl w-full overflow-hidden shadow-[0_0_40px_rgba(16,185,129,0.15)] my-auto"
      >
        {/* Top Tactical Header */}
        <div className="px-4 sm:px-6 py-3.5 bg-gradient-to-r from-emerald-950/70 via-zinc-900 to-zinc-950 border-b border-zinc-800 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <div className="p-1.5 rounded-lg bg-emerald-500/15 border border-emerald-500/40 text-emerald-400">
              <BookOpen className="w-4 h-4" />
            </div>
            <div>
              <h2 className="font-mono font-bold text-xs sm:text-sm uppercase tracking-wider text-zinc-100">
                Operator Field Guide <span className="text-emerald-400">//</span> AI Teaching &amp; Biometrics
              </h2>
              <p className="text-[10px] sm:text-xs font-mono text-zinc-400">
                5-second cheat sheet for fixing mislabeled Kill-Feed cards without polluting Face ID
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-zinc-400 hover:text-white bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 rounded-md p-1.5 cursor-pointer"
            title="Close (Esc)"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-4 sm:p-6 space-y-5 max-h-[82vh] overflow-y-auto font-mono">
          {/* Section 1: Mislabeled Card Decision Matrix */}
          <div>
            <div className="flex items-center justify-between mb-2.5">
              <span className="text-[11px] font-bold uppercase tracking-wider text-emerald-400 flex items-center gap-1.5">
                <Eye className="w-3.5 h-3.5" />
                1. Decision Matrix — &ldquo;A Kill-Feed Card Is Mislabeled. What Do I Do?&rdquo;
              </span>
              <span className="hidden sm:inline-block text-[10px] text-zinc-500">
                SFace = Facial Geometry · Qwen2.5-VL = Visual Context
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
              {/* Card 1: Stranger called Family */}
              <div className="rounded-lg bg-red-950/25 border border-red-500/40 p-3 space-y-1.5">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-red-600 text-white uppercase">
                    🔴 ALWAYS FIX THIS
                  </span>
                  <span className="text-[10px] text-red-300 font-bold">FALSE FAMILY MATCH</span>
                </div>
                <div className="text-xs font-bold text-zinc-100">
                  Stranger wrongly labeled <span className="text-emerald-300">ISAAC</span> or{' '}
                  <span className="text-emerald-300">JENNIFER</span> (front OR from behind)
                </div>
                <div className="text-[11px] bg-zinc-950/90 border border-red-500/30 rounded px-2 py-1 text-amber-300 font-bold">
                  👉 Click: 🎓 TEACH → PERSON → 👤 STRANGER
                </div>
                <p className="text-[10px] text-zinc-400 leading-relaxed">
                  Fixes the card, restores <span className="text-red-300">🚨 BREACH</span> if in your tripwire, and{' '}
                  <span className="text-zinc-200">deletes any bad camera vector</span> that caused the false match.
                </p>
              </div>

              {/* Card 2: Family from behind or Trash-Bin False Carrier */}
              <div className="rounded-lg bg-amber-950/20 border border-amber-500/40 p-3 space-y-1.5">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-amber-500 text-black uppercase">
                    🟡 BACK OF HEAD / CHORES
                  </span>
                  <span className="text-[10px] text-amber-300 font-bold">NO FACE / TRASH BINS</span>
                </div>
                <div className="text-xs font-bold text-zinc-100">
                  You/family from behind labeled <span className="text-zinc-300">person</span> (or{' '}
                  <span className="text-amber-300">📦 AMAZON</span> with blue trash bins)
                </div>
                <div className="text-[11px] bg-zinc-950/90 border border-amber-500/30 rounded px-2 py-1 text-zinc-200 font-bold">
                  👉 Ignore &ldquo;person&rdquo;  |  or  🎓 TEACH → ✅ ISAAC / JENNIFER
                </div>
                <p className="text-[10px] text-zinc-400 leading-relaxed">
                  Clears <span className="text-red-300">🚨 BREACH</span> / false <span className="text-amber-300">AMAZON</span>{' '}
                  on the card while <span className="text-zinc-200">blocking bad face vectors</span> when no sharp face is visible
                  (never use <span className="text-sky-300">👤 ID FACE</span> from behind).
                </p>
              </div>

              {/* Card 3: Delivery / Landscapers */}
              <div className="rounded-lg bg-emerald-950/25 border border-emerald-500/40 p-3 space-y-1.5">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-emerald-600 text-white uppercase">
                    🟢 DO TEACH (ANY ANGLE)
                  </span>
                  <span className="text-[10px] text-emerald-300 font-bold">CARRIERS &amp; CREWS</span>
                </div>
                <div className="text-xs font-bold text-zinc-100">
                  Amazon / UPS / FedEx / Landscapers labeled <span className="text-zinc-300">person</span> (even from
                  behind!)
                </div>
                <div className="text-[11px] bg-zinc-950/90 border border-emerald-500/30 rounded px-2 py-1 text-emerald-300 font-bold">
                  👉 Click: 🎓 TEACH → DELIVERY → 📦 AMAZON / 🔇 LANDSCAPERS
                </div>
                <p className="text-[10px] text-zinc-400 leading-relaxed">
                  Writes a camera rule for <span className="text-zinc-200">Qwen2.5-VL</span>. Vests, boxes, trucks, and
                  mowers <span className="text-emerald-300">DO</span> work from behind.
                </p>
              </div>

              {/* Card 4: Animals & Clear Family Faces */}
              <div className="rounded-lg bg-sky-950/25 border border-sky-500/40 p-3 space-y-1.5">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-sky-600 text-white uppercase">
                    🟢 DO TEACH
                  </span>
                  <span className="text-[10px] text-sky-300 font-bold">ANIMALS &amp; CLEAR FACES</span>
                </div>
                <div className="text-xs font-bold text-zinc-100">
                  Cat called <span className="text-zinc-300">dog</span> — OR family member{' '}
                  <span className="text-sky-300">facing camera</span> called <span className="text-zinc-300">person</span>
                </div>
                <div className="text-[11px] bg-zinc-950/90 border border-sky-500/30 rounded px-2 py-1 text-sky-300 font-bold">
                  👉 Click: 🎓 TEACH → 🐱 CAT  |  or  ✅ ISAAC / JENNIFER
                </div>
                <p className="text-[10px] text-zinc-400 leading-relaxed">
                  Animals update VLM prompt memory; clear front-facing family shots safely tag the card (and only save
                  vectors if eyes/nose/mouth are sharp).
                </p>
              </div>
            </div>
          </div>

          {/* Section 2: Which Tool Does What */}
          <div>
            <span className="text-[11px] font-bold uppercase tracking-wider text-amber-300 flex items-center gap-1.5 mb-2.5">
              <Cpu className="w-3.5 h-3.5" />
              2. Know Your 3 Buttons — Enrollment vs. Correction
            </span>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
              <div className="rounded-lg bg-zinc-900/90 border border-emerald-500/30 p-3 space-y-1">
                <div className="flex items-center gap-1.5 text-emerald-300 text-xs font-bold">
                  <Upload className="w-3.5 h-3.5 shrink-0" />
                  FACE GALLERY → UPLOAD
                </div>
                <div className="text-[10px] text-emerald-400 font-bold uppercase">Best for Family Whitelist</div>
                <p className="text-[10px] text-zinc-400 leading-relaxed">
                  Upload <span className="text-zinc-100 font-bold">1–2 crisp phone photos</span> per family member
                  (good lighting, no hat/sunglasses). Beats 15 blurry driveway captures.
                </p>
              </div>

              <div className="rounded-lg bg-zinc-900/90 border border-amber-500/30 p-3 space-y-1">
                <div className="flex items-center gap-1.5 text-amber-300 text-xs font-bold">
                  <Check className="w-3.5 h-3.5 shrink-0" />
                  🎓 TEACH (Card Button)
                </div>
                <div className="text-[10px] text-amber-400 font-bold uppercase">Best for Fixing Wrong Labels</div>
                <p className="text-[10px] text-zinc-400 leading-relaxed">
                  1-click presets to cure false matches (<span className="text-zinc-100">👤 STRANGER</span>) or teach
                  VLM context (<span className="text-zinc-100">🐱 CAT</span>, <span className="text-zinc-100">📦 AMAZON</span>,{' '}
                  <span className="text-zinc-100">🔇 LANDSCAPERS</span>).
                </p>
              </div>

              <div className="rounded-lg bg-zinc-900/90 border border-sky-500/30 p-3 space-y-1">
                <div className="flex items-center gap-1.5 text-sky-300 text-xs font-bold">
                  <UserCheck className="w-3.5 h-3.5 shrink-0" />
                  👤 ID FACE (Card Button)
                </div>
                <div className="text-[10px] text-sky-400 font-bold uppercase">New Custom Names / Watchlist</div>
                <p className="text-[10px] text-zinc-400 leading-relaxed">
                  Use only when a <span className="text-zinc-100 font-bold">clear face</span> is visible and you want
                  to name a regular (<span className="text-zinc-100">MAILMAN</span>, <span className="text-zinc-100">NEIGHBOR</span>)
                  or flag <span className="text-red-300 font-bold">🚨 WATCHLIST</span>.
                </p>
              </div>
            </div>
          </div>

          {/* Bottom Police Lineup Callout */}
          <div className="rounded-lg bg-zinc-900 border border-zinc-700 px-3.5 py-2.5 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div className="flex items-start sm:items-center gap-2.5">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5 sm:mt-0" />
              <div className="text-[11px] text-zinc-200">
                <span className="font-bold text-amber-300">THE POLICE LINEUP RULE:</span> If a stranger couldn&apos;t
                pick someone out of a lineup using <span className="underline">only</span> that tiny cropped thumbnail,
                don&apos;t enroll it. <span className="text-emerald-300 font-bold">2 sharp photos &gt; 10 blurry ones.</span>
              </div>
            </div>
            <button
              onClick={onClose}
              className="px-3 py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold flex items-center justify-center gap-1.5 shrink-0 cursor-pointer"
            >
              <ShieldCheck className="w-3.5 h-3.5" />
              GOT IT
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
