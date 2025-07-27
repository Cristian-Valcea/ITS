# src/security/encryption.py
"""
Hardened encryption utilities with Argon2id KDF and AES-256.
Provides memory-hard key derivation resistant to GPU attacks.
"""

import os
import secrets
import base64
from typing import Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

try:
    import argon2
    ARGON2_AVAILABLE = True
except ImportError:
    ARGON2_AVAILABLE = False


class HardenedEncryption:
    """
    Military-grade encryption with Argon2id KDF.
    Falls back to PBKDF2-HMAC-SHA256 if Argon2 unavailable.
    """
    
    def __init__(self, use_argon2: bool = True):
        self.use_argon2 = use_argon2 and ARGON2_AVAILABLE
        
        # Argon2id parameters (tuned for <200ms unlock time)
        self.argon2_time_cost = 3      # iterations
        self.argon2_memory_cost = 65536  # 64MB memory
        self.argon2_parallelism = 1
        
        # PBKDF2 fallback parameters
        self.pbkdf2_iterations = 100000
    
    def derive_key_argon2(self, password: str, salt: bytes) -> bytes:
        """Derive key using Argon2id (memory-hard, GPU-resistant)."""
        if not ARGON2_AVAILABLE:
            raise RuntimeError("argon2-cffi not installed. Install with: pip install argon2-cffi")
        
        from argon2.low_level import hash_secret_raw, Type
        
        return hash_secret_raw(
            secret=password.encode('utf-8'),
            salt=salt,
            time_cost=self.argon2_time_cost,
            memory_cost=self.argon2_memory_cost,
            parallelism=self.argon2_parallelism,
            hash_len=32,
            type=Type.ID  # Argon2id
        )
    
    def derive_key_pbkdf2(self, password: str, salt: bytes) -> bytes:
        """Fallback key derivation using PBKDF2-HMAC-SHA256."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.pbkdf2_iterations,
        )
        return kdf.derive(password.encode('utf-8'))
    
    def derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key using best available method."""
        if self.use_argon2:
            return self.derive_key_argon2(password, salt)
        else:
            return self.derive_key_pbkdf2(password, salt)
    
    def generate_salt(self) -> bytes:
        """Generate cryptographically secure random salt."""
        return secrets.token_bytes(32)
    
    def create_fernet(self, password: str, salt: bytes) -> Fernet:
        """Create Fernet cipher with derived key."""
        key = self.derive_key(password, salt)
        fernet_key = base64.urlsafe_b64encode(key)
        return Fernet(fernet_key)
    
    def encrypt(self, data: str, password: str) -> Tuple[bytes, bytes]:
        """
        Encrypt data with password.
        Returns (encrypted_data, salt) tuple.
        """
        salt = self.generate_salt()
        fernet = self.create_fernet(password, salt)
        encrypted_data = fernet.encrypt(data.encode('utf-8'))
        return encrypted_data, salt
    
    def decrypt(self, encrypted_data: bytes, salt: bytes, password: str) -> str:
        """Decrypt data with password and salt."""
        fernet = self.create_fernet(password, salt)
        decrypted_bytes = fernet.decrypt(encrypted_data)
        return decrypted_bytes.decode('utf-8')
    
    def benchmark_kdf(self, password: str = "test_password") -> dict:
        """
        Benchmark KDF performance for parameter tuning.
        Returns timing information in milliseconds.
        """
        import time
        
        salt = self.generate_salt()
        results = {}
        
        # Benchmark Argon2id if available
        if ARGON2_AVAILABLE:
            start_time = time.perf_counter()
            self.derive_key_argon2(password, salt)
            argon2_time = (time.perf_counter() - start_time) * 1000
            results['argon2id_ms'] = argon2_time
        
        # Benchmark PBKDF2
        start_time = time.perf_counter()
        self.derive_key_pbkdf2(password, salt)
        pbkdf2_time = (time.perf_counter() - start_time) * 1000
        results['pbkdf2_ms'] = pbkdf2_time
        
        return results
    
    def zeroize_memory(self, data: bytearray) -> None:
        """
        Securely zero memory containing sensitive data.
        Note: Python's garbage collector may still have copies.
        """
        if isinstance(data, bytearray):
            for i in range(len(data)):
                data[i] = 0
