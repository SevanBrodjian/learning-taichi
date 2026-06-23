import { mediaUrl } from "../api.js";

// Rendered playback of a run. Video for now; a p5/canvas particle scrubber is a natural upgrade.
export default function SimPlayer({ media }) {
  if (media?.video) {
    return (
      <video className="sim-video" src={mediaUrl(media.video)} controls loop muted playsInline />
    );
  }
  if (media?.frames_dir) {
    return <div className="muted sim-video placeholder">Frame sequence available (viewer TBD).</div>;
  }
  return <div className="muted sim-video placeholder">No rendered video for this run.</div>;
}
