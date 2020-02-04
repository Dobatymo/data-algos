# Cryptography

## Diffie–Hellman key exchange
- also called: 'DH'
- https://en.wikipedia.org/wiki/Diffie%E2%80%93Hellman_key_exchange
- is a: 'Key-agreement protocol'
- based on: 'Diffie–Hellman problem', 'Discrete logarithm problem'
- implemented in: 'C OpenSSL'
- domain: 'Public-key cryptography'
- licence: 'Public domain' (since 1997)

## X3DH
- also called: 'Extended Triple Diffie-Hellman'
- https://signal.org/docs/specifications/x3dh/
- based on: 'Diffie–Hellman key exchange'
- is a: 'Key-agreement protocol'

## Elliptic-curve Diffie–Hellman
- also called: 'ECDH'
- https://en.wikipedia.org/wiki/Elliptic-curve_Diffie%E2%80%93Hellman
- is a: 'Key-agreement protocol'
- based on: 'Elliptic-curve cryptography'
- domain: 'Public-key cryptography'
- variant of: 'Diffie–Hellman key exchange'
- implemented in (libraries): 'C OpenSSL'
- implemented by (applications): 'Line'
- used by (protocols): 'Signal Protocol'

## Curve25519
- paper: 'Curve25519: new Diffie-Hellman speed records' (2006)
- https://en.wikipedia.org/wiki/Curve25519
- is a: 'Elliptic curve'
- for use with: 'Elliptic-curve Diffie–Hellman'
- implemented in (libraries): 'C NaCl'
- used by (protocols): 'OMEMO', 'Secure Shell', 'Transport Layer Security', 'Tox'
- implemented by (applications): 'Wire', 'Threema'

## RSA
- also called: 'Rivest–Shamir–Adleman'
- https://en.wikipedia.org/wiki/RSA_(cryptosystem)
- depends on computational hardness of: 'RSA problem'
- domain: 'Public-key cryptography'
- licence: 'Public domain' (since 2000)

## Asynchronous Ratcheting Tree
- also called: 'ART'
- paper: 'On Ends-to-Ends Encryption: Asynchronous Group Messaging with Strong Security Guarantees' (2017)
- https://github.com/facebookresearch/asynchronousratchetingtree
- is a: 'Cryptographic protocol'

## TreeKEM
- also called: 'MLS ratchet tree'
- based on: 'Asynchronous Ratcheting Tree'

## Double Ratchet algorithm
- also called: 'Axolotl Ratchet'
- whitepaper: 'The Double Ratchet Algorithm' (2016)
- https://en.wikipedia.org/wiki/Double_Ratchet_Algorithm
- https://signal.org/docs/specifications/doubleratchet/
- is a: 'key management algorithm'
- based on: 'X3DH', 'Key derivation function'
- implemented by: 'Signal Protocol', 'Signal', 'WhatsApp'
- implemented in: 'C++ Olm', 'python-doubleratchet'
- applications: 'End-to-end encryption', 'Instant messaging'
- features: 'Post-compromise security'

## Square
- also called: 'SQUARE'
- https://en.wikipedia.org/wiki/Square_(cipher)
- paper: 'The Block Cipher SQUARE' (1997)
- is a: 'Block cipher'
- bits: '128'
- implemented in: 'Crypto++'

## Advanced Encryption Standard
- also called: 'AES', 'Rijndael'
- https://en.wikipedia.org/wiki/Advanced_Encryption_Standard
- is a: 'Block cipher', 'Symmetric-key algorithm'
- bits: '128', '192', '256'
- based on: 'Square'
- implemented in: 'Crypto++', 'C OpenSSL', 'C NaCl'

## Salsa20
- https://en.wikipedia.org/wiki/Salsa20
- paper: 'The Salsa20 Family of Stream Ciphers' (2008)
- is a: 'Stream cipher', 'Symmetric-key algorithm'
- implemented in: 'Crypto++', 'C NaCl'

## ChaCha20
- based on: 'Salsa20'
- is a: 'Stream cipher', 'Symmetric-key algorithm'
- RFC: 7539, 8439
- implemented in: 'Crypto++', 'C OpenSSL'

## Poly1305
- is a: 'Message authentication code'
- RFC: 7539, 8439
- implemented in: 'Crypto++', 'C OpenSSL', 'C NaCl'

-- key derivation function

## Lyra2
- https://en.wikipedia.org/wiki/Lyra2
- is a: 'Key derivation function'
- applications: 'Proof of work', 'Cryptocurrency'

## scrypt
- https://en.wikipedia.org/wiki/Scrypt
- is a: 'Key derivation function'
- applications: 'Proof of work', 'Cryptocurrency'
- RFC: 7914

## PBKDF2
- also called: 'Password-Based Key Derivation Function 2'
- https://en.wikipedia.org/wiki/PBKDF2
- is a: 'Key derivation function'
- recommended by: 'PKCS #5'

## Argon2
- https://en.wikipedia.org/wiki/Argon2
- is a: 'Key derivation function'
- winner of: 'Password Hashing Competition' (2015)

-- block cipher modes of operation

## ECB
- also called: 'Electronic Codebook'
- susceptible to: 'replay attacks'
- encryption parallelizable: Yes
- decryption parallelizable: Yes
- random read access: Yes
- don't use

## CBC
- also called: 'Cipher-block chaining'
- patent: 'Message verification and transmission error detection by block chaining' (1976)
- https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#CBC
- encryption parallelizable: No
- decryption parallelizable: Yes
- random read access: Yes
- susceptible to: 'padding oracle attacks', 'malleability attacks'

## CFB
- also called: 'Cipher Feedback'
- encryption parallelizable: No
- decryption parallelizable: Yes
- random read access: Yes

## CTR
- also called: 'Counter'
- paper: 'Privacy and authentication: An introduction to cryptography' (1979)
- encryption parallelizable: Yes
- decryption parallelizable: Yes
- random read access: Yes
- susceptible to: 'malleability attacks'

## XTS
- also called: 'XEX-based tweaked-codebook mode with ciphertext stealing'
- paper: '1619-2007 - IEEE Standard for Cryptographic Protection of Data on Block-Oriented Storage Devices' (2008)
- https://en.wikipedia.org/wiki/Disk_encryption_theory#XEX-based_tweaked-codebook_mode_with_ciphertext_stealing_(XTS)
- applications: 'Full disk encryption'
- authentication: no
- implemented by (applications): 'VeraCrypt', 'Android', 'dm-crypt'

## CMC
- also called: 'CBC–mask–CBC'

## EME
- also called: 'ECB–mask–ECB', 'ECB-Mix-ECB', 'Encrypt-Mix-Encrypt'
- paper: 'A Parallelizable Enciphering Mode' (2003)
- patent: 'Block cipher mode of operation for constructing a wide-blocksize block cipher from a conventional block cipher' (2003)
- wide-block encryption

## EME*
- also called: 'EME2'
- paper: 'EME*: Extending EME to Handle Arbitrary-Length Messages with Associated Data' (2004)
- wide-block encryption
- refinement of: 'EME'

## XCB
- also called: 'Extended codebook mode'
- paper: 'The Extended Codebook (XCB) Mode of Operation' (2004)
- analysis paper: 'The Security of the Extended Codebook (XCB) Mode of Operation' (2007)

-- Authenticated block cipher modes of operation

## GCM
- also called: 'Galios/Counter Mode'
- https://en.wikipedia.org/wiki/Galois/Counter_Mode

## CCM
- also called: 'Counter with CBC-MAC'
- https://en.wikipedia.org/wiki/CCM_mode

## CCM*
- variant of: 'CCM'
- used by: 'ZigBee'

## EAX
- also called: 'Encrypt-then-authenticate-then-translate mode'
- https://en.wikipedia.org/wiki/EAX_mode

## OCB
- also called: 'Offset codebook mode'
- https://en.wikipedia.org/wiki/OCB_mode
- patented

## CWC
- also called: 'Carter–Wegman + CTR mode'
- https://en.wikipedia.org/wiki/CWC_mode
