# Uncategorized protocols

## Garbled circuit
- also called: 'Yao's garbled circuits protocol'
- solves: 'Secure multi-party computation'

## Privacy-Preserving Scalar Product Protocol
- also called: 'PPSPP'
- solves: 'Two-way privacy-preserving distance calculation'
- paper: 'Secure two-party distance computation protocol based on privacy homomorphism and scalar product in wireless sensor networks' (2016) <https://doi.org/10.1109/TST.2016.7536716>

## Homomorphic-Encryption-based Euclidean Distance Protocol
- also called: 'HEEDP'
- solves: 'Two-way privacy-preserving distance calculation'
- paper: 'Secure two-party distance computation protocol based on privacy homomorphism and scalar product in wireless sensor networks' (2016) <https://doi.org/10.1109/TST.2016.7536716>

## Spatial cloaking
- solves: 'One-way privacy-preserving distance calculation'
- paper: 'Anonymous Usage of Location-Based Services Through Spatial and Temporal Cloaking' (2003) <https://doi.org/10.1145/1066116.1189037>
- paper: 'Evaluating the Privacy Guarantees of Location Proximity Services' (2017) <https://doi.org/10.1145/3007209>
- idea: 'round coordinates, not distances'

# Protocols (distributed algorithms, not simple network protocols)

## Identity-Based Privacy-Protected Access Control Filter
- also called: 'IPACF'
- is a: 'Protocol'
- applications: 'denial-of-service defense'

# Congestion control / network scheduling algorithms

## Additive increase/multiplicative decrease
- also called: 'AIMD'
- https://en.wikipedia.org/wiki/Additive_increase/multiplicative_decrease
- is a: 'congestion control algorithm'
- applications: 'TCP congestion control'
- used by: 'TCP', 'UDT'

## Weighted round robin
- https://en.wikipedia.org/wiki/Weighted_round_robin

## Deficit round robin
- paper: 'Efficient fair queueing using deficit round robin' (1995)
- https://en.wikipedia.org/wiki/Deficit_round_robin
- is a: 'scheduling algorithm'
- properties: 'fair'

## Token bucket
- https://en.wikipedia.org/wiki/Token_bucket
- is a: 'scheduling algorithm'
- applications: 'Traffic policing', 'Traffic shaping'

## Leaky bucket
- https://en.wikipedia.org/wiki/Leaky_bucket
- applications: 'Traffic policing', 'Traffic shaping'

## Generic cell rate algorithm
- also called: 'GCRA'
- https://en.wikipedia.org/wiki/Generic_cell_rate_algorithm
- is a: 'scheduling algorithm'
- based on: 'Leaky bucket'
- used for: 'Asynchronous transfer mode' networks

## LEDBAT
- also called: 'Low Extra Delay Background Transport'
- https://en.wikipedia.org/wiki/LEDBAT
- RFC: 6817
- is a: 'congestion control algorithm'

## SACK
- also called: 'Selective acknowledgment'
- RFC: 2018

# Cryptographic communication protocols

## Secure Shell
- also called: 'SSH'
- https://en.wikipedia.org/wiki/Secure_Shell
- implemented in (applications): 'OpenSSH'

## Secure Sockets Layer
- also called: 'SSL'
- deprecated
- OSI model: 'Presentation layer'

## Transport Layer Security
- also called: 'TLS'
- https://en.wikipedia.org/wiki/Transport_Layer_Security
- superseeds: 'Secure Sockets Layer'
- implemented in (libraries): 'OpenSSL', 'LibreSSL', 'Network Security Services', 'Secure Channel', 'GnuTLS'
- OSI model: 'Presentation layer'

## Datagram Transport Layer Security
- also called: 'DTLS'
- https://en.wikipedia.org/wiki/Datagram_Transport_Layer_Security
- variant of: 'Transport Layer Security'

## OpenVPN
- https://en.wikipedia.org/wiki/OpenVPN
- applications: 'Virtual private network', 'Tunneling'
- implemented in (applications): 'OpenVPN', 'SoftEther VPN'

## IPSec IKEv2

## WireGuard
- whitepaper: 'WireGuard: Next Generation Kernel Network Tunnel' (2017)
- https://en.wikipedia.org/wiki/WireGuard
- applications: 'Virtual private network', 'Tunneling'
- implemented in (applications): 'WireGuard'
- uses: 'ChaCha20', 'Curve25519', 'BLAKE2s', 'SipHash24', 'HKDF'
- note: currently has privacy problems due to static ip assignment

## Silent Circle Instant Messaging Protocol
- also called: 'SCIMP'
- https://en.wikipedia.org/wiki/Silent_Circle_Instant_Messaging_Protocol
- features: 'Encryption', 'Forward secrecy', 'Authentication'
- implemented by (applications): 'Silent Text'
- deprecated

## Off-the-Record Messaging
- also called: 'OTR'
- https://en.wikipedia.org/wiki/Off-the-Record_Messaging
- is a: 'Cryptographic protocol'
- uses: 'Diffie–Hellman key exchange', 'SHA-1', 'Socialist millionaire protocol'
- features: 'Forward secrecy', 'Deniable authentication'
- implemented by (applications): 'ChatSecure', 'Psi', 'Jitsi'

## Signal Protocol
- also called: 'TextSecure Protocol'
- https://en.wikipedia.org/wiki/Signal_Protocol
- is a: 'Cryptographic protocol'
- uses: 'Double Ratchet algorithm', 'Curve25519', 'Advanced Encryption Standard', 'HMAC-SHA256'
- implemented by (applications): 'Signal'
- applications: 'End-to-end encryption'
- implemented in (libraries): 'libsignal-protocol-c', 'libsignal-protocol-java', 'libsignal-protocol-javascript'

## Proteus
- https://en.wikipedia.org/wiki/Wire_(software)
- based on: 'Signal Protocol'
- implemented by (applications): 'Wire'

## OMEMO
- also called: 'OMEMO Multi-End Message and Object Encryption', 'XEP-0384'
- https://en.wikipedia.org/wiki/OMEMO
- extension to: 'XMPP'
- applications: 'End-to-end encryption'
- features: 'Forward secrecy', 'Deniable authentication'
- uses: 'Double Ratchet algorithm'
- implemented by (applications): 'ChatSecure', 'Psi', 'Conversations'
- implemented in (libraries): 'C libomemo', 'python-omemo'

## Messaging Layer Security
- also called: 'MLS'
- https://messaginglayersecurity.rocks/
- https://en.wikipedia.org/wiki/Messaging_Layer_Security
- properties: 'draft'
- uses: 'TreeKEM'
- primitives: 'SHA-256', 'Curve25519', 'AES-128-GCM'

## MTProto
- https://core.telegram.org/mtproto
- implemented by (applications): 'Telegram'

## Tox
- https://en.wikipedia.org/wiki/Tox_(protocol)
- applications: 'End-to-end encryption'
- uses: 'Curve25519', 'Poly1305'

## Bitmessage
- https://en.wikipedia.org/wiki/Bitmessage
- whitepaper: 'Bitmessage: A Peer-to-Peer Message Authentication and Delivery System' (2012)

## Jami
- also called: 'GNU Ring'
- https://en.wikipedia.org/wiki/Jami_(software)
- uses: 'DHT', 'SIP'
- implemented by (applications): 'Jami'

## Ricochet
- https://en.wikipedia.org/wiki/Ricochet_(software)
- applications: 'Instant messaging'
- uses: 'TOR'

## Matrix
- https://en.wikipedia.org/wiki/Matrix_(protocol)
- https://matrix.org/
- consists of: 'Olm', 'Megolm'
- implemented by (applications): 'Riot'

## Mobile CoWPI
- also called: 'Mobile Conversations With Privacy and Integrity'
- paper: 'End-to-End Secure Mobile Group Messaging with Conversation Integrity and Deniability' (2018)

# Transport protocols

## TCP
- also called: 'Transmission Control Protocol'
- https://en.wikipedia.org/wiki/Transmission_Control_Protocol
- usually layered on: 'IP'
- features: 'stream based', 'congestion control'
- uses: 'SACK', 'Nagle's algorithm'

## UDP
- also called: 'User Datagram Protocol'
- https://en.wikipedia.org/wiki/User_Datagram_Protocol
- usually layered on: 'IP'

## SCTP
- also called: 'Stream Control Transmission Protocol'
- https://en.wikipedia.org/wiki/Stream_Control_Transmission_Protocol
- RFC: 4960
- features: 'multihoming', 'message based', 'congestion control'
- usually layered on: 'UDP', 'IP'

## QUIC
- also called: 'Quick UDP Internet Connections'
- https://en.wikipedia.org/wiki/QUIC
- RFC: 9000
- usually layered on: 'UDP'
- features: 'FEC', 'congestion control', 'multiplexing without head-of-line blocking'

# Streaming protocols

## RTMFP
- also called: 'Real-Time Media Flow Protocol'
- https://en.wikipedia.org/wiki/Real-Time_Media_Flow_Protocol
- RFC: 7016
- usually layered on: 'UDP'

## Automatic repeat request
- also called: 'ARQ'
- https://en.wikipedia.org/wiki/Automatic_repeat_request

## MSS
- also called: 'Microsoft Smooth Streaming'
- uses: 'HTTP'

## HDS
- also called: 'HTTP Dynamic Streaming'
- uses: 'HTTP'

## HLS
- also called: 'HTTP Live Streaming'
- https://en.wikipedia.org/wiki/HTTP_Live_Streaming
- uses: 'HTTP'
- RFC: 8216
- container: 'MPEG-TS'

## DASH
- also called: 'Dynamic Adaptive Streaming over HTTP', 'MPEG-DASH'
- uses: 'TCP', 'HTTP'
- codec-agnostic
- Adaptive bitrate streaming
- container: 'MP4'

## RTMP
- also called: 'Real-Time Messaging Protocol'
- https://en.wikipedia.org/wiki/Real-Time_Messaging_Protocol
- usually layered on: 'TCP'

## RTP
- also called: 'Real-time Transport Protocol'
- RFC: 3550
- https://en.wikipedia.org/wiki/Real-time_Transport_Protocol
- uses: 'UDP'
- used with: 'RTCP'
- see also: 'SRTP'
- features: 'Content agnostic'
- standardized payload formats: 'Flexible Forward Error Correction (FEC)'

## RTSP
- also called: 'Real Time Streaming Protocol'
- https://en.wikipedia.org/wiki/Real_Time_Streaming_Protocol
- uses: 'RTP', 'RTCP'

## SRTP
- also called: 'Secure Real-time Transport Protocol'
- https://en.wikipedia.org/wiki/Secure_Real-time_Transport_Protocol
- see also: 'RTP'

## RTCP
- also called: 'RTP Control Protocol'
- https://en.wikipedia.org/wiki/RTP_Control_Protocol

## WebRTC
- also called: 'Web Real-Time Communication'
- https://en.wikipedia.org/wiki/WebRTC
- https://webrtc.org/
- Open Source
- 'Data Channel' uses 'SCTP', 'DTLS', 'UDP'
- 'Media Transport' uses 'SRTP'

## FTL
- also called: 'Faster Than Light'
- uses: 'UDP', 'RTP'
- proprietary

## Secure Reliable Transport
- also called: 'SRT'
- https://en.wikipedia.org/wiki/Secure_Reliable_Transport
- https://www.haivision.com/products/srt-secure-reliable-transport/
- Open Source
- features: 'FEC', 'AES encryption', 'Content agnostic'
- based on: 'UDT'
- supposed to be a replacement for: 'RTMP'

## Reliable Internet Stream Transport
- also called: 'RIST'
- https://en.wikipedia.org/wiki/Reliable_Internet_Stream_Transport
- features: 'ARQ', 'FEC', 'Bitmask NACK', 'Range NACK'
- uses: 'RTCP', 'SMPTE-2022 Forward Error Correction', 'RTP', 'DTLS'

# Communication protocols

## XMPP
- also called: 'Extensible Messaging and Presence Protocol', 'Jabber'
- https://en.wikipedia.org/wiki/XMPP
- applications: 'Instant messaging', 'Presence information'
- properties: 'server-to-server', 'federated'
- OSI model: 'Application layer'
- standardized by: 'IETF'

## Internet Message Access Protocol
- also called: 'IMAP'
- properties: 'text based'
- OSI model: 'Application layer'
- RFC: 3501

## Simple Mail Transfer Protocol
- also called: 'SMTP'
- properties: 'text-based'
- OSI model: 'Application layer'

## Push-IMAP
- https://en.wikipedia.org/wiki/Push-IMAP
- extension to: 'Internet Message Access Protocol'
- OSI model: 'Application layer'

## File Transfer Protocol
- also called: 'FTP'
- https://en.wikipedia.org/wiki/File_Transfer_Protocol
- is a: 'Communication protocol'
- applications: 'File transfer'
- OSI model: 'Application layer'

## File eXchange Protocol
- also called: 'FXP'
- https://en.wikipedia.org/wiki/File_eXchange_Protocol
- extension to: 'File Transfer Protocol'
- is a: 'Communication protocol'
- properties: 'server-to-server'
- applications: 'File transfer'
- OSI model: 'Application layer'

## Simple File Transfer Protocol
- also called: 'SFTP'
- https://en.wikipedia.org/wiki/File_Transfer_Protocol#Simple_File_Transfer_Protocol
- applications: 'File transfer'
- RFC: 913
- outdated

## SSH File Transfer Protocol
- also called: 'SFTP'
- https://en.wikipedia.org/wiki/SSH_File_Transfer_Protocol
- applications: 'File transfer'
- extension to: 'Secure Shell'
- implemented by (applications): 'OpenSSH'

## Fast and Secure Protocol
- also called: 'FASP'
- https://en.wikipedia.org/wiki/Fast_and_Secure_Protocol
- properties: 'patented', 'proprietary'
- applications: 'File transfer'

## UDP-based Data Transfer Protocol
- also called: 'UDT'
- paper: 'UDT: UDP-based data transfer for high-speed wide area networks' (2007)
- https://en.wikipedia.org/wiki/UDP-based_Data_Transfer_Protocol
- http://udt.sourceforge.net/
- applications: 'File transfer'
- based on: 'UDP'
- uses: 'Additive increase/multiplicative decrease'
- implemented by (applications): 'GridFTP'

## Tsunami UDP Protocol
- https://en.wikipedia.org/wiki/Tsunami_UDP_Protocol
- https://sourceforge.net/projects/tsunami-udp/
- applications: 'File transfer'
- based on: 'UDP'

## Instant Messaging and Presence Protocol
- also called: 'IMPP'
- https://en.wikipedia.org/wiki/Instant_Messaging_and_Presence_Protocol
- outdated
- OSI model: 'Application layer'

## SILC
- also called: 'Secure Internet Live Conferencing protocol'
- https://en.wikipedia.org/wiki/SILC_(protocol)
- is a: 'Communication protocol'
- properties: 'server-to-server', 'federated'
- OSI model: 'Application layer'

## REsource LOcation And Discovery
- also called: 'RELOAD'
- https://en.wikipedia.org/wiki/REsource_LOcation_And_Discovery_Framing
- analysis paper: 'Next generation protocol for P2P SIP communication' (2011)
- analysis paper: 'Analysis of relod.net, a basic implementation of the RELOAD protocol for peer-to-peer networks'
- is a: 'Signaling protocol'
- properties: 'p2p'

## Session Initiation Protocol
- also called: 'SIP'
- https://en.wikipedia.org/wiki/Session_Initiation_Protocol
- is a: 'Signaling protocol'
- applications: 'Text messaging', 'Voice over IP'
- OSI model: 'Application layer'

## Zephyr
- https://en.wikipedia.org/wiki/Zephyr_(protocol)
- applications: 'Instant messaging'
- OSI model: 'Application layer'
- outdated

## SOCKS
- https://en.wikipedia.org/wiki/SOCKS
- OSI model: 'Session layer'

## ActivityPub
- https://en.wikipedia.org/wiki/ActivityPub
- https://www.w3.org/TR/activitypub/
- properties: 'server-to-server', 'federated'

## WebSub
- also called: 'PubSubHubbub'
- https://en.wikipedia.org/wiki/WebSub
- properties: 'Publish–subscribe'
- standardized by: 'W3C'
- based on: 'HTTP'

## WebSocket
- https://en.wikipedia.org/wiki/WebSocket
- is a: 'Communication protocol'
- properties: 'full-duplex'

## Advanced Message Queuing Protocol
- also called: 'AMQP'
- https://en.wikipedia.org/wiki/Advanced_Message_Queuing_Protocol
- is a: 'Message queue', 'Message-oriented middleware'
- properties: 'binary/wire-level'
- standardized by: 'OASIS', 'ISO'
- implemented by: 'RabbitMQ', 'Apache ActiveMQ'

## Streaming Text Oriented Messaging Protocol
- https://en.wikipedia.org/wiki/Streaming_Text_Oriented_Messaging_Protocol
- also called: 'STOMP'
- properties: 'text-based'
- is a: 'Message queue'
- implemented by: 'RabbitMQ', 'Apache ActiveMQ'

## Message Queuing Telemetry Transport
- https://en.wikipedia.org/wiki/MQTT
- also called: 'MQTT'
- is a: 'Message queue'
- standardized by: 'OASIS', 'ISO'
- implemented by: 'RabbitMQ'

## BitTorrent
- https://en.wikipedia.org/wiki/Comparison_of_file_transfer_protocols
- applications: 'File transfer'
- properties: 'p2p'

## Micro Transport Protocol
- also called: 'µTP', 'uTP'
- https://en.wikipedia.org/wiki/Micro_Transport_Protocol
- variant of: 'BitTorrent'
- based on: 'UDP'
- properties: 'p2p'
- implemented in (libraries): 'libutp'
- implemented by (applications): 'µTorrent', 'qBittorrent'
- alternative to: 'TCP'
- uses: 'LEDBAT'
