import pandas as pd
from ml.csv_loader import read_csv_smart

def add_team_info_to_players():
    """Add teamID column to players.csv based on player names and team mappings"""
    
    # Load existing data
    players_df = read_csv_smart('data/players.csv')
    teams_df = read_csv_smart('data/teams.csv')
    
    print(f"Loaded {len(players_df)} players and {len(teams_df)} teams")
    
    # Create a mapping of player names to team IDs based on known players
    # This is a simplified mapping - you can expand this based on your data
    player_team_mapping = {
        # Manchester United players (teamID: 89)
        "Wayne Rooney": 89,
        "Juan Mata": 89,
        "Memphis Depay": 89,
        "Ashley Young": 89,
        "Ander Herrera": 89,
        "Antonio Valencia": 89,
        "Bastian Schweinsteiger": 89,
        "Sergio Romero": 89,
        "Matteo Darmian": 89,
        "Daley Blind": 89,
        "Chris Smalling": 89,
        "Luke Shaw": 89,
        "Morgan Schneiderlin": 89,
        "Michael Carrick": 89,
        
        # Tottenham players (teamID: 82)
        "Harry Kane": 82,
        "Christian Eriksen": 82,
        "Dele Alli": 82,
        "Erik Lamela": 82,
        "Kyle Walker": 82,
        "Toby Alderweireld": 82,
        "Jan Vertonghen": 82,
        "Ben Davies": 82,
        "Nabil Bentaleb": 82,
        "Eric Dier": 82,
        "Mousa Dembélé": 82,
        "Nacer Chadli": 82,
        "Ryan Mason": 82,
        "Michel Vorm": 82,
        
        # Arsenal players (teamID: 83) - would need to be added
        # Liverpool players (teamID: 87) - would need to be added
        # Chelsea players (teamID: 80) - would need to be added
        
        # Bournemouth players (teamID: 73)
        "Artur Boruc": 73,
        "Simon Francis": 73,
        "Steve Cook": 73,
        "Tommy Elphick": 73,
        "Charlie Daniels": 73,
        "Matt Ritchie": 73,
        "Andrew Surman": 73,
        "Dan Gosling": 73,
        "Marc Pugh": 73,
        "Joshua King": 73,
        "Callum Wilson": 73,
        "Yann Kermorgant": 73,
        "Eunan O'Kane": 73,
        "Max Gradel": 73,
        
        # Aston Villa players (teamID: 71)
        "Brad Guzan": 71,
        "Leandro Bacuna": 71,
        "Micah Richards": 71,
        "Ciaran Clark": 71,
        "Jordan Amavi": 71,
        "Idrissa Gueye": 71,
        "Ashley Westwood": 71,
        "Jordan Veretout": 71,
        "Jordan Ayew": 71,
        "Gabriel Agbonlahor": 71,
        "Scott Sinclair": 71,
        "Carlos Sánchez": 71,
        "Kieran Richardson": 71,
        "Rudy Gestede": 71,
        
        # Everton players (teamID: 72)
        "Tim Howard": 72,
        "Seamus Coleman": 72,
        "Phil Jagielka": 72,
        "John Stones": 72,
        "Brendan Galloway": 72,
        "James McCarthy": 72,
        "Gareth Barry": 72,
        "Kevin Mirallas": 72,
        "Ross Barkley": 72,
        "Tom Cleverley": 72,
        "Romelu Lukaku": 72,
        "Arouna Koné": 72,
        "Bryan Oviedo": 72,
        "Steven Naismith": 72,
        
        # Watford players (teamID: 90)
        "Heurelho Gomes": 90,
        "Nyom": 90,
        "Sebastian Prödl": 90,
        "Craig Cathcart": 90,
        "José Holebas": 90,
        "Ikechi Anya": 90,
        "Valon Behrami": 90,
        "Etienne Capoue": 90,
        "Miguel Layún": 90,
        "Jurado": 90,
        "Troy Deeney": 90,
        "Ben Watson": 90,
        "Juan Carlos Paredes": 90,
        "Odion Ighalo": 90,
        
        # Leicester players (teamID: 75)
        "Kasper Schmeichel": 75,
        "Ritchie de Laet": 75,
        "Robert Huth": 75,
        "Wes Morgan": 75,
        "Jeffrey Schlupp": 75,
        "Marc Albrighton": 75,
        "Andy King": 75,
        "Daniel Drinkwater": 75,
        "Riyad Mahrez": 75,
        "Jamie Vardy": 75,
        "Shinji Okazaki": 75,
        "N'Golo Kanté": 75,
        "Christian Fuchs": 75,
        "Yohan Benalouane": 75,
        "Costel Pantilimon": 75,
    }
    
    # Add teamID column to players dataframe
    players_df['teamID'] = players_df['name'].map(player_team_mapping)
    
    # Count how many players got team assignments
    assigned_count = players_df['teamID'].notna().sum()
    print(f"Assigned team IDs to {assigned_count} out of {len(players_df)} players")
    
    # For players without team assignments, assign random teams for demo purposes
    # In a real scenario, you'd want to manually map these or get the data from a proper source
    unassigned_mask = players_df['teamID'].isna()
    if unassigned_mask.sum() > 0:
        # Get list of team IDs
        team_ids = teams_df['teamID'].tolist()
        # Assign random teams to unassigned players
        import random
        random.seed(42)  # For reproducible results
        random_teams = random.choices(team_ids, k=unassigned_mask.sum())
        players_df.loc[unassigned_mask, 'teamID'] = random_teams
        print(f"Assigned random teams to {unassigned_mask.sum()} remaining players")
    
    # Add some sample performance data for demo purposes
    import random
    random.seed(42)
    
    # Add rating, goals, assists, minutes columns with realistic values
    players_df['rating'] = [random.randint(65, 90) for _ in range(len(players_df))]
    players_df['goals'] = [random.randint(0, 25) for _ in range(len(players_df))]
    players_df['assists'] = [random.randint(0, 15) for _ in range(len(players_df))]
    players_df['minutes'] = [random.randint(500, 3000) for _ in range(len(players_df))]
    
    # Save the updated players data
    output_path = 'data/players_with_teams.csv'
    players_df.to_csv(output_path, index=False)
    print(f"Saved updated players data to {output_path}")
    
    # Show sample of the updated data
    print("\nSample of updated players data:")
    print(players_df[['playerID', 'name', 'teamID', 'rating', 'goals', 'assists', 'minutes']].head(10))
    
    return output_path

if __name__ == "__main__":
    add_team_info_to_players()
