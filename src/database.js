const { Pool } = require('pg');
const bcrypt = require('bcrypt');
const path = require('path');
const ini = require('ini');
const fs = require('fs');

// Get config path from command line arguments or use default
const configPath = process.argv.find(arg => arg.startsWith('--config='))
  ? process.argv.find(arg => arg.startsWith('--config=')).split('=')[1]
  : './conf/config.ini';

console.log(`Database loading configuration from: ${configPath}`);
const config = ini.parse(fs.readFileSync(path.resolve(configPath), 'utf-8'));

// PostgreSQL connection configuration
const pgConfig = {
  host: config.database['database-host'],
  port: config.database['database-port'],
  user: config.database['database-user'],
  password: config.database['database-password'],
  database: config.database['database-name'],
  max: config.database.max_connections || 20,
  ssl: config.database.ssl === 'true'
};

console.log(`Connecting to PostgreSQL at ${pgConfig.host}:${pgConfig.port}`);

// Create a pool variable that we'll assign either a real pool or a mock
let pool;

// Try to create a real PostgreSQL pool
try {
  pool = new Pool(pgConfig);
  
  // Test the connection
  pool.connect((err, client, release) => {
    if (err) {
      console.error('Error connecting to PostgreSQL:', err);
      console.log('Creating mock database for UI testing...');
      
      // Create a mock pool for UI testing
      pool = createMockPool();
    } else {
      console.log(`Connected to PostgreSQL database: ${pgConfig.database}`);
      release();
    }
  });
} catch (error) {
  console.error('Error creating PostgreSQL pool:', error);
  console.log('Creating mock database for UI testing...');
  
  // Create a mock pool for UI testing
  pool = createMockPool();
}

// Function to create a mock pool for UI testing
function createMockPool() {
  console.log('Using mock database for UI testing');
  
  // In-memory storage for mock data
  const mockData = {
    users: [
      { id: 1, username: 'admin', password: '$2b$10$rM7tU0Zl3DP1I1qZRVSCX.j.IK0k1uI5XbmJz.g9r7CQKVu6JSGcK', email: 'admin@local.host', role: 'admin', name: 'Administrator' }
    ],
    chat_sessions: [],
    messages: [],
    ollama_settings: [
      { id: 1, host: 'localhost', port: 11434, default_model: '' }
    ]
  };
  
  return {
    query: async (text, params) => {
      console.log('Mock DB Query:', text, params);
      
      // Handle different types of queries
      if (text.includes('SELECT * FROM users')) {
        return { rows: mockData.users, rowCount: mockData.users.length };
      }
      
      if (text.includes('SELECT * FROM ollama_settings')) {
        return { rows: mockData.ollama_settings, rowCount: mockData.ollama_settings.length };
      }
      
      if (text.includes('INSERT INTO ollama_settings')) {
        // Don't insert if we already have settings
        if (mockData.ollama_settings.length === 0) {
          const newSettings = {
            id: 1,
            host: params[0],
            port: params[1],
            default_model: params[2]
          };
          mockData.ollama_settings.push(newSettings);
        }
        return { rows: [mockData.ollama_settings[0]], rowCount: 1 };
      }
      
      if (text.includes('SELECT id, title, created_at, last_message_timestamp, is_active FROM chat_sessions')) {
        return { rows: mockData.chat_sessions, rowCount: mockData.chat_sessions.length };
      }
      
      if (text.includes('INSERT INTO chat_sessions')) {
        const newSession = {
          id: params[0],
          user_id: params[1],
          title: params[2],
          created_at: new Date(),
          last_message_timestamp: new Date(),
          is_active: true
        };
        mockData.chat_sessions.push(newSession);
        return { rows: [newSession], rowCount: 1 };
      }
      
      if (text.includes('INSERT INTO messages')) {
        const newMessage = {
          id: Math.floor(Math.random() * 1000000),
          user_id: params[0],
          message: params[1],
          response: params[2],
          session_id: params[3],
          timestamp: new Date()
        };
        mockData.messages.push(newMessage);
        return { rows: [newMessage], rowCount: 1 };
      }
      
      // Default response for unhandled queries
      return { rows: [], rowCount: 0 };
    },
    connect: (callback) => {
      callback(null, { query: async () => ({ rows: [] }) }, () => {});
    },
    end: () => {
      console.log('Mock database connection closed');
    }
  };
}

// Helper function to convert SQLite '?' placeholders to PostgreSQL '$1' style placeholders
function convertPlaceholders(text) {
  let paramIndex = 0;
  return text.replace(/\?/g, () => `$${++paramIndex}`);
}

// Create a database interface object that mimics the SQLite interface
// to minimize changes in existing code
const db = {
  // Query executor function
  async query(text, params) {
    try {
      const convertedText = convertPlaceholders(text);
      const result = await pool.query(convertedText, params);
      return result;
    } catch (error) {
      console.error('Database query error:', error);
      throw error;
    }
  },

  // For code compatibility with SQLite, we'll provide similar methods
  // but adapted for PostgreSQL
  prepare(text) {
    const convertedText = convertPlaceholders(text);

    return {
      get: async (...params) => {
        try {
          const result = await pool.query(convertedText, params);
          return result.rows[0] || null;
        } catch (error) {
          console.error(`Error in db.prepare().get() with query '${convertedText}':`, error);
          throw error;
        }
      },
      all: async (...params) => {
        try {
          const result = await pool.query(convertedText, params);
          return result.rows;
        } catch (error) {
          console.error(`Error in db.prepare().all() with query '${convertedText}':`, error);
          throw error;
        }
      },
      run: async (...params) => {
        try {
          const result = await pool.query(convertedText, params);
          // For compatibility with better-sqlite3, we add lastInsertRowid
          // This approximation works only for single inserts
          let lastInsertRowid = null;
          if (result.rows && result.rows[0] && result.rows[0].id) {
            lastInsertRowid = result.rows[0].id;
          }

          return {
            changes: result.rowCount || 0,
            lastInsertRowid: lastInsertRowid
          };
        } catch (error) {
          console.error(`Error in db.prepare().run() with query '${convertedText}':`, error);
          throw error;
        }
      }
    };
  },

  // Execute raw SQL
  async exec(text) {
    try {
      await pool.query(text);
    } catch (error) {
      console.error('SQL execution error:', error);
      throw error;
    }
  }
};

async function initializeDatabase() {
  console.log('Checking database connection');

  try {
    // Check if we're using the mock database
    const isMockDatabase = pool.query.toString().includes('Mock DB Query');
    
    if (isMockDatabase) {
      console.log('Using mock database, skipping migrations and setup');
      return;
    }
    
    // Run migrations for real database
    await runMigrations();

    // Run document table creation script
    try {
      // Check if the script exists before attempting to run it
      const fixScriptPath = path.resolve(__dirname, 'scripts/sql/fix_documents_table.sql');
      if (fs.existsSync(fixScriptPath)) {
        await runMigration('scripts/sql/fix_documents_table.sql');
        console.log('Documents table creation/fix completed');
      } else {
        console.log('Fix documents table script not found, skipping');
      }
    } catch (error) {
      console.error('Error creating/fixing documents table:', error.message);
    }

    // Add name column to users table if it doesn't exist
    try {
      console.log('Checking if name column exists in users table...');
      await pool.query(`
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = 'users'
            AND column_name = 'name'
          ) THEN
            ALTER TABLE users ADD COLUMN name VARCHAR(100);
            RAISE NOTICE 'Added name column to users table';
          ELSE
            RAISE NOTICE 'name column already exists in users table';
          END IF;
        END $$;
      `);
      console.log('Name column check completed');
    } catch (error) {
      console.error('Error checking/adding name column:', error.message);
    }

    // Check if admin user exists, if not create default admin
    const userQuery = await pool.query("SELECT * FROM users WHERE username = $1", [config.admin.default_username]);

    if (userQuery.rows.length === 0) {
      const hashedPassword = bcrypt.hashSync(config.admin.default_password, 10);

      // Modified query to include name field
      await pool.query(
        "INSERT INTO users (username, password, email, role, name) VALUES ($1, $2, $3, $4, $5) RETURNING id",
        [config.admin.default_username, hashedPassword, config.admin.default_email || 'admin@local.host', 'admin', 'Administrator']
      );
      console.log('Default admin user created');
    } else {
      // Update admin user to add name if it's missing
      if (userQuery.rows[0].name === null || userQuery.rows[0].name === undefined) {
        await pool.query(
          "UPDATE users SET name = $1 WHERE username = $2",
          ['Administrator', config.admin.default_username]
        );
        console.log('Updated admin user with name');
      }
      console.log('Admin user already exists');
    }
  } catch (error) {
    console.error('Database initialization error:', error);
    // Don't exit the process, just log the error
    console.error('More details:', error.message);
  }
}

process.on('exit', () => {
  // Close the pool when the application exits
  pool.end();
  console.log('Database connection pool closed');
});

/**
 * Runs a migration script file
 */
const runMigration = async (scriptPath) => {
  try {
    // Attempt to find the script in multiple locations
    let resolvedPath;
    const possiblePaths = [
      path.resolve(__dirname, '..', scriptPath), // Try relative to project root
      path.resolve(__dirname, scriptPath),       // Try relative to src
      path.resolve(scriptPath)                   // Try absolute path
    ];
    
    for (const testPath of possiblePaths) {
      if (fs.existsSync(testPath)) {
        resolvedPath = testPath;
        break;
      }
    }
    
    if (!resolvedPath) {
      console.log(`Migration script not found: ${scriptPath}`);
      return false;
    }
    
    const scriptContent = fs.readFileSync(resolvedPath, 'utf8');
    await db.query(scriptContent);
    console.log(`Migration script ${scriptPath} executed successfully`);
    return true;
  } catch (error) {
    console.error(`Error executing migration script ${scriptPath}:`, error);
    return false;
  }
};

/**
 * Runs database migrations for model_id column
 */
const runModelIdMigration = async () => {
  try {
    // Check if we're using the mock database
    const isMockDatabase = pool.query.toString().includes('Mock DB Query');
    
    if (isMockDatabase) {
      console.log('Using mock database, skipping model ID migration');
      return true;
    }
    
    await runMigration('scripts/sql/add_model_id_to_ai_models.sql');
    console.log('Model ID migration completed');
    return true;
  } catch (error) {
    console.error('Error running model_id migration:', error);
    return false;
  }
};

/**
 * Runs all migrations in order
 */
const runMigrations = async () => {
  try {
    console.log('Running database migrations...');
    
    // Check if we're using the mock database
    const isMockDatabase = pool.query.toString().includes('Mock DB Query');
    
    if (isMockDatabase) {
      console.log('Using mock database, skipping migrations');
      return;
    }

    // Get all migration files
    const migrationsDir = path.join(__dirname, 'migrations');
    if (!fs.existsSync(migrationsDir)) {
      console.log('No migrations directory found, skipping migrations');
      return;
    }

    const migrationFiles = fs.readdirSync(migrationsDir)
      .filter(file => file.endsWith('.js'))
      .sort(); // Sort to ensure migrations run in order

    // Get list of applied migrations
    let appliedMigrationNames = [];
    try {
      // Create migrations table if it doesn't exist
      await pool.query(`
        CREATE TABLE IF NOT EXISTS schema_migrations (
          id SERIAL PRIMARY KEY,
          version VARCHAR(255) NOT NULL UNIQUE,
          applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
          description TEXT,
          name VARCHAR(255)
        )
      `);
    } catch (error) {
      console.error('Error creating migrations table:', error);
      throw error;
    }

    // Get list of applied migrations
    try {
      // Try to get migrations by name first (for compatibility)
      let appliedMigrations = await pool.query('SELECT name FROM schema_migrations WHERE name IS NOT NULL');
      
      if (appliedMigrations.rows.length === 0) {
        // If no migrations found by name, try with version
        appliedMigrations = await pool.query('SELECT version FROM schema_migrations WHERE version IS NOT NULL');
        appliedMigrationNames = appliedMigrations.rows.map(row => row.version);
      } else {
        appliedMigrationNames = appliedMigrations.rows.map(row => row.name);
      }
      
      console.log('Applied migrations:', appliedMigrationNames);
    } catch (error) {
      console.error('Error getting applied migrations, assuming none applied:', error.message);
      // Continue with empty list - will try to apply all migrations
    }

    // Run each migration that hasn't been applied yet
    for (const file of migrationFiles) {
      if (!appliedMigrationNames.includes(file)) {
        console.log(`Running migration: ${file}`);
        const migration = require(path.join(migrationsDir, file));

        try {
          await migration.up();

          // Record that the migration has been applied
          // Extract version number from file name (e.g., 005_add_documents_table.js -> 005)
          const versionMatch = file.match(/^(\d+)_/);
          const version = versionMatch ? versionMatch[1] : file;
          const description = file.replace(/\.js$/, '').replace(/^\d+_/, '').replace(/_/g, ' ');
          
          // Insert with both name and version for compatibility
          await pool.query(
            'INSERT INTO schema_migrations (name, version, description) VALUES ($1, $2, $3)',
            [file, version, description]
          );
          
          console.log(`Migration ${file} completed successfully`);
        } catch (error) {
          console.error(`Error running migration ${file}:`, error);
          throw error;
        }
      } else {
        console.log(`Migration ${file} already applied, skipping`);
      }
    }

    console.log('All migrations completed successfully');
  } catch (error) {
    console.error('Error running migrations:', error);
    throw error;
  }
};

// Export functions
module.exports = {
  db,
  pool, // Export the pool for direct access if needed
  initializeDatabase,
  runMigration,
  runModelIdMigration,
  runMigrations
};