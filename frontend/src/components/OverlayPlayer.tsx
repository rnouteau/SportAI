type OverlayPlayerProps = {
  url: string;
};

export default function OverlayPlayer({ url }: OverlayPlayerProps) {
  return (
    <div className="overflow-hidden rounded-xl bg-black shadow-sm">
      <video
        src={url}
        controls
        preload="metadata"
        playsInline
        className="aspect-video w-full"
      >
        Your browser cannot play this video
      </video>
    </div>
  );
}
