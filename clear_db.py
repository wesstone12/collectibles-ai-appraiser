import sqlite3
import os

def clear_database():
    try:
        # Delete the database file
        db_path = 'output/pokemon_cards.db'
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Successfully deleted {db_path}")
        
        # Also delete any Excel files in output directory
        for file in os.listdir('output'):
            if file.endswith('.xlsx'):
                os.remove(os.path.join('output', file))
                print(f"Successfully deleted output/{file}")
                
        print("\nDatabase and Excel files cleared. Ready for fresh start!")
        
    except Exception as e:
        print(f"Error clearing database: {str(e)}")

if __name__ == "__main__":
    clear_database() 