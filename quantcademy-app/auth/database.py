"""
Database Authentication Module
Supports Supabase (recommended) or SQLite (local development)

Supabase Setup:
1. Create account at supabase.com
2. Create new project
3. Get URL and anon key from Settings > API
4. Add to .env:
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-anon-key

SQLite (Local):
- No setup needed, uses local file
- Good for development/testing
"""

import os
import hashlib
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

# Try to load environment variables
try:
    from dotenv import load_dotenv
    # Force reload to ensure env vars are loaded
    load_dotenv(Path(__file__).parent.parent / ".env", override=True)
except ImportError:
    pass

# Debug: Print which backend will be used
_debug_url = os.environ.get('SUPABASE_URL', '')
_debug_key = os.environ.get('SUPABASE_KEY', '')
print(f"[Auth] SUPABASE_URL set: {bool(_debug_url)}, SUPABASE_KEY set: {bool(_debug_key)}")


# =============================================================================
# Database Backend Selection
# =============================================================================

SUPABASE_AVAILABLE = False
SQLITE_AVAILABLE = True

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
    print("[Auth] supabase package: LOADED")
except ImportError as e:
    print(f"[Auth] supabase package: FAILED TO LOAD - {e}")


def get_database_backend() -> str:
    """Determine which database backend to use."""
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_KEY')
    
    print(f"[Auth] get_database_backend: SUPABASE_AVAILABLE={SUPABASE_AVAILABLE}, url={bool(supabase_url)}, key={bool(supabase_key)}")
    
    if SUPABASE_AVAILABLE and supabase_url and supabase_key:
        return 'supabase'
    return 'sqlite'


# =============================================================================
# Password Hashing
# =============================================================================

def hash_password(password: str, salt: str = None) -> tuple:
    """Hash a password with salt."""
    if salt is None:
        salt = secrets.token_hex(32)
    
    # Use PBKDF2 with SHA256
    hashed = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000  # iterations
    ).hex()
    
    return hashed, salt


def verify_password(password: str, hashed: str, salt: str) -> bool:
    """Verify a password against its hash."""
    new_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(new_hash, hashed)


# =============================================================================
# Supabase Backend
# =============================================================================

class SupabaseAuth:
    """Authentication using Supabase."""
    
    def __init__(self):
        url = os.environ.get('SUPABASE_URL')
        key = os.environ.get('SUPABASE_KEY')
        
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY required")
        
        self.client: Client = create_client(url, key)
    
    def create_user(self, email: str, password: str, name: str) -> Dict[str, Any]:
        """Create a new user."""
        try:
            # Check if user exists
            existing = self.client.table('users').select('*').eq('email', email).execute()
            if existing.data:
                return {'success': False, 'error': 'Email already registered'}
            
            # Hash password
            hashed, salt = hash_password(password)
            
            # Insert user
            result = self.client.table('users').insert({
                'email': email,
                'name': name,
                'password_hash': hashed,
                'password_salt': salt,
                'created_at': datetime.utcnow().isoformat(),
                'last_login': None
            }).execute()
            
            if result.data:
                return {
                    'success': True,
                    'user': {
                        'id': result.data[0]['id'],
                        'email': email,
                        'name': name
                    }
                }
            
            return {'success': False, 'error': 'Failed to create user'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def authenticate(self, email: str, password: str) -> Dict[str, Any]:
        """Authenticate a user."""
        try:
            # Get user
            result = self.client.table('users').select('*').eq('email', email).execute()
            
            if not result.data:
                return {'success': False, 'error': 'User not found'}
            
            user = result.data[0]
            
            # Verify password
            if not verify_password(password, user['password_hash'], user['password_salt']):
                return {'success': False, 'error': 'Invalid password'}
            
            # Update last login
            self.client.table('users').update({
                'last_login': datetime.utcnow().isoformat()
            }).eq('id', user['id']).execute()
            
            return {
                'success': True,
                'user': {
                    'id': user['id'],
                    'email': user['email'],
                    'name': user['name']
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user by ID."""
        try:
            result = self.client.table('users').select('*').eq('id', user_id).execute()
            if result.data:
                user = result.data[0]
                return {
                    'id': user['id'],
                    'email': user['email'],
                    'name': user['name']
                }
            return None
        except:
            return None
    
    def update_user(self, user_id: str, data: Dict) -> bool:
        """Update user data."""
        try:
            self.client.table('users').update(data).eq('id', user_id).execute()
            return True
        except:
            return False
    
    def save_user_progress(self, user_id: str, progress: Dict) -> bool:
        """Save user learning progress."""
        try:
            # Check if progress exists
            existing = self.client.table('user_progress').select('*').eq('user_id', user_id).execute()
            
            if existing.data:
                self.client.table('user_progress').update({
                    'progress_data': progress,
                    'updated_at': datetime.utcnow().isoformat()
                }).eq('user_id', user_id).execute()
            else:
                self.client.table('user_progress').insert({
                    'user_id': user_id,
                    'progress_data': progress,
                    'created_at': datetime.utcnow().isoformat(),
                    'updated_at': datetime.utcnow().isoformat()
                }).execute()
            
            return True
        except:
            return False
    
    def get_user_progress(self, user_id: str) -> Optional[Dict]:
        """Get user learning progress."""
        try:
            result = self.client.table('user_progress').select('*').eq('user_id', user_id).execute()
            if result.data:
                return result.data[0]['progress_data']
            return None
        except:
            return None


# =============================================================================
# SQLite Backend (Local Development)
# =============================================================================

class SQLiteAuth:
    """Authentication using local SQLite database."""
    
    def __init__(self, db_path: str = None):
        import sqlite3
        
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "users.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables."""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_login TEXT
            )
        ''')
        
        # User progress table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                progress_data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _get_conn(self):
        import sqlite3
        return sqlite3.connect(self.db_path)
    
    def create_user(self, email: str, password: str, name: str) -> Dict[str, Any]:
        """Create a new user."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
            if cursor.fetchone():
                conn.close()
                return {'success': False, 'error': 'Email already registered'}
            
            # Hash password
            hashed, salt = hash_password(password)
            
            # Insert user
            cursor.execute('''
                INSERT INTO users (email, name, password_hash, password_salt, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (email, name, hashed, salt, datetime.utcnow().isoformat()))
            
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return {
                'success': True,
                'user': {
                    'id': user_id,
                    'email': email,
                    'name': name
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def authenticate(self, email: str, password: str) -> Dict[str, Any]:
        """Authenticate a user."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, email, name, password_hash, password_salt
                FROM users WHERE email = ?
            ''', (email,))
            
            row = cursor.fetchone()
            
            if not row:
                conn.close()
                return {'success': False, 'error': 'User not found'}
            
            user_id, email, name, hashed, salt = row
            
            if not verify_password(password, hashed, salt):
                conn.close()
                return {'success': False, 'error': 'Invalid password'}
            
            # Update last login
            cursor.execute('''
                UPDATE users SET last_login = ? WHERE id = ?
            ''', (datetime.utcnow().isoformat(), user_id))
            
            conn.commit()
            conn.close()
            
            return {
                'success': True,
                'user': {
                    'id': user_id,
                    'email': email,
                    'name': name
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_user(self, user_id: int) -> Optional[Dict]:
        """Get user by ID."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            cursor.execute('SELECT id, email, name FROM users WHERE id = ?', (user_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {'id': row[0], 'email': row[1], 'name': row[2]}
            return None
        except:
            return None
    
    def update_user(self, user_id: int, data: Dict) -> bool:
        """Update user data."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            updates = ', '.join([f'{k} = ?' for k in data.keys()])
            values = list(data.values()) + [user_id]
            
            cursor.execute(f'UPDATE users SET {updates} WHERE id = ?', values)
            conn.commit()
            conn.close()
            return True
        except:
            return False
    
    def save_user_progress(self, user_id: int, progress: Dict) -> bool:
        """Save user learning progress."""
        import json
        
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            progress_json = json.dumps(progress)
            now = datetime.utcnow().isoformat()
            
            cursor.execute('SELECT id FROM user_progress WHERE user_id = ?', (user_id,))
            
            if cursor.fetchone():
                cursor.execute('''
                    UPDATE user_progress SET progress_data = ?, updated_at = ?
                    WHERE user_id = ?
                ''', (progress_json, now, user_id))
            else:
                cursor.execute('''
                    INSERT INTO user_progress (user_id, progress_data, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, progress_json, now, now))
            
            conn.commit()
            conn.close()
            return True
        except:
            return False
    
    def get_user_progress(self, user_id: int) -> Optional[Dict]:
        """Get user learning progress."""
        import json
        
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            cursor.execute('SELECT progress_data FROM user_progress WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return json.loads(row[0])
            return None
        except:
            return None


# =============================================================================
# Factory Function
# =============================================================================

def get_auth_backend():
    """Get the appropriate authentication backend."""
    backend = get_database_backend()
    
    if backend == 'supabase':
        return SupabaseAuth()
    else:
        return SQLiteAuth()


# =============================================================================
# Convenience Functions
# =============================================================================

_auth_instance = None

def get_auth():
    """Get singleton auth instance."""
    global _auth_instance
    if _auth_instance is None:
        backend = get_database_backend()
        print(f"[Auth] Initializing {backend} backend")
        _auth_instance = get_auth_backend()
    return _auth_instance


def reset_auth():
    """Reset the auth instance (useful for testing)."""
    global _auth_instance
    _auth_instance = None


def signup(email: str, password: str, name: str) -> Dict[str, Any]:
    """Create a new user account."""
    try:
        result = get_auth().create_user(email, password, name)
        print(f"[Auth] Signup result for {email}: {result}")
        return result
    except Exception as e:
        print(f"[Auth] Signup error: {e}")
        return {'success': False, 'error': str(e)}


def login(email: str, password: str) -> Dict[str, Any]:
    """Authenticate a user."""
    try:
        result = get_auth().authenticate(email, password)
        print(f"[Auth] Login result for {email}: success={result.get('success')}")
        return result
    except Exception as e:
        print(f"[Auth] Login error: {e}")
        return {'success': False, 'error': str(e)}


def get_user(user_id) -> Optional[Dict]:
    """Get user by ID."""
    return get_auth().get_user(user_id)


def save_progress(user_id, progress: Dict) -> bool:
    """Save user learning progress."""
    return get_auth().save_user_progress(user_id, progress)


def load_progress(user_id) -> Optional[Dict]:
    """Load user learning progress."""
    return get_auth().get_user_progress(user_id)
