import { useRef, useState } from "react";

// Clean autoplay loop with on-demand controls, shared by task results and training-page embeds.
// Native `controls` on iOS keeps a big overlay up over an autoplaying video until tapped, which is
// obstructive when flicking through content. Instead the video plays clean and a minimal control bar
// (play/pause + scrub + fullscreen) appears on hover (desktop) or tap (touch). The play/pause state is
// explicit, so a paused video stays paused (no native-controls / autoplay tug-of-war).
export default function VideoPlayer({ src, className = "task-video" }) {
  const ref = useRef(null);
  const [playing, setPlaying] = useState(true);
  const [cur, setCur] = useState(0);
  const [dur, setDur] = useState(0);
  const [show, setShow] = useState(false);
  const toggle = () => {
    const v = ref.current;
    if (v) (v.paused ? v.play() : v.pause());
  };
  const fullscreen = () => {
    const v = ref.current;
    if (!v) return;
    if (v.requestFullscreen) v.requestFullscreen();
    else if (v.webkitEnterFullscreen) v.webkitEnterFullscreen(); // iOS Safari (native fullscreen player)
  };
  return (
    <div className="vid" onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)} onClick={() => setShow((s) => !s)}>
      <video
        ref={ref}
        className={className}
        src={src}
        autoPlay loop muted playsInline preload="auto"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onTimeUpdate={(e) => setCur(e.currentTarget.currentTime)}
        onLoadedMetadata={(e) => setDur(e.currentTarget.duration || 0)}
      />
      <div className={`vid-controls ${show ? "show" : ""}`} onClick={(e) => e.stopPropagation()}>
        <button className="vid-btn" onClick={toggle} aria-label={playing ? "pause" : "play"}>{playing ? "❚❚" : "▶"}</button>
        <input
          className="vid-seek" type="range" min="0" max={dur || 0} step="0.01" value={cur}
          onChange={(e) => { if (ref.current) ref.current.currentTime = parseFloat(e.target.value); }}
        />
        <button className="vid-btn" onClick={fullscreen} aria-label="fullscreen">⛶</button>
      </div>
    </div>
  );
}
