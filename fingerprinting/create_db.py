import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

def create_db():
    """
    Creates the database.
    """
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cur = conn.cursor()
        cur.execute("""CREATE TABLE songs (
            song_id INT PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            artist VARCHAR(255) NOT NULL,
            duration_ms INT NOT NULL,
            link VARCHAR(255)
            );"""
        )
        cur.execute("""CREATE TABLE fingerprints (
            hash VARCHAR(255) NOT NULL,
            offset_time_ms INT NOT NULL,
            song_id INT NOT NULL,
            PRIMARY KEY (hash, song_id, offset_time_ms),
            FOREIGN KEY (song_id) REFERENCES songs(song_id)
            );"""
        )
        conn.commit()
        cur.close()
        conn.close()
        print("Database and tables created successfully")
    except psycopg2.Error as e:
        print(f"Error creating database: {e}")
        conn.rollback()
        cur.close()
        conn.close()
        return False
    return True

if __name__ == '__main__':
    create_db()