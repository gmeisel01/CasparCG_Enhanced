CasparCG Enhanced

A build-ready fork of CasparCG Server maintained by ifelseWare.
This fork exists to provide a single download that includes three improvements to CasparCG that are currently under review for inclusion in the main project. You can build this fork today without waiting for upstream merges.
What's included
Screen Consumer

Fixed GL viewport initialization for non-standard canvas sizes (ultra-wide, LED walls, multi-display spanning)
Extended <aspect-ratio> config supporting width:height ratio strings and decimal values
<always-on-top>, <borderless>, <brightness-boost>, <saturation-boost>, <enable-mipmaps> options
casparcg.config documentation and examples for custom video modes and spanning configurations

OAL System Audio Consumer

Video-scheduled audio dispatch eliminates drift against the video clock

PortAudio Consumer

New consumer supporting ASIO and JACK multi-channel audio output with configurable channel count and latency tuning

Building
Follow the standard CasparCG build instructions. All three features are included in the working branch and require no additional configuration to enable.
