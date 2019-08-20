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
