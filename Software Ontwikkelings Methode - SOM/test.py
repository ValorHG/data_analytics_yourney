import google.generativeai as genai
import pandas as pd
import unittest
from unittest.mock import patch, MagicMock
from Lab01 import BoardGameMechanicsAnalyzer
from io import StringIO

class TestBoardGameMechanicsAnalyzer(unittest.TestCase):

    # Test 1: Eliminating missing values
    def test_eliminating_missing_values(self):
        mock_data = StringIO("""ID;Name;Year Published;Min Players;Max Players;Play Time;Complexity;BGG Rank;Average Rating;Total Ratings;Mechanics;Categories
174430;Gloomhaven;2017;1;4;120;14;42055;8.79;1;3.86;68323;Action Queue;Strategy Games
161936;Pandemic Legacy: Season 1;2015;2;4;60;13;41643;8.61;2;2.84;65294;Action Points;Strategy Games
224517;;;2;4;;;;8.66;3;3.91;28785;Hand Management;Strategy Games
167791;Terraforming Mars;2016;1;5;120;12;64864;8.43;4;3.24;87099;Card Drafting;Strategy Games
233078;Twilight Imperium: Fourth Edition;2017;3;6;480;14;;;;;Action Drafting;Strategy Games
""")

        # API key placeholder
        api_key = "dummy_api_key"

        # Create an instance of BoardGameMechanicsAnalyzer with the mock data
        analyzer = BoardGameMechanicsAnalyzer(api_key, mock_data)

        # Assert that only rows with complete data remain
        self.assertEqual(len(analyzer.Data), 3)  # Only 3 rows with valid data remain
        # Assert there are no null values in the cleaned dataset
        self.assertFalse(analyzer.Data.isnull().values.any())

    # Test 2: Accuracy calculation

@patch("Lab01.BoardGameMechanicsAnalyzer.chat_with_gemini")
def test_accuracy_calculation(self, mock_chat_with_gemini):
        # Mock response from the query_gemini method (simulating the API response)
        mock_response = {
            "correct_mechanics": ["Action Queue", "Campaign / Battle Card Driven"],
            "total_predictions": ["Action Queue", "Action Retrieval", "Campaign / Battle Card Driven"]
        }
        
        # Setting up the mock return value
        mock_chat_with_gemini.return_value = mock_response
        
        # Use StringIO to simulate file input
        mock_data = """ID;Name;Year Published;Min Players;Max Players;Play Time;Complexity;BGG Rank;Average Rating;Total Ratings;Mechanics;Categories
174430;Gloomhaven;2017;1;4;120;14;42055;8.79;1;3.86;68323;Action Queue;Strategy Games
161936;Pandemic Legacy: Season 1;2015;2;4;60;13;41643;8.61;2;2.84;65294;Action Points;Strategy Games
167791;Terraforming Mars;2016;1;5;120;12;64864;8.43;4;3.24;87099;Card Drafting;Strategy Games
"""

        # Use StringIO to simulate the file-like object for pandas
        from io import StringIO
        mock_data_io = StringIO(mock_data)
        
        # Creating an instance of BoardGameMechanicsAnalyzer with mock_data_io
        analyzer = BoardGameMechanicsAnalyzer("dummy_api_key", mock_data_io)
        
        game_name = "Gloomhaven"
        year_published = 2017
        
        # Call the method that calculates accuracy
        accuracy = analyzer.calculate_accuracy(game_name, year_published)

        # Check that the accuracy is calculated as the number of correct mechanics divided by total predictions
        expected_accuracy = len(mock_response["correct_mechanics"]) / len(mock_response["total_predictions"])
        self.assertEqual(accuracy, expected_accuracy)

    # Test 3: Invalid and valid input

@patch('builtins.input', side_effect=["2", "Gloomhaven", "2017", "Unknown Game", "2021"])  # Mocking user input
@patch('builtins.print')  # Mock print to verify output

def test_multiple_games(self, mock_print, mock_input):
      
        # Mock response from Gemini
        mock_chat_with_gemini = MagicMock(return_value="Mocked Gemini Response")
        
        # Simulating a small BGG dataset
        mock_data = StringIO("""ID;Name;Year Published;Min Players;Max Players;Play Time;Complexity;BGG Rank;Average Rating;Total Ratings;Mechanics;Categories
174430;Gloomhaven;2017;1;4;120;14;42055;8.79;1;3.86;68323;Action Queue;Strategy Games
""")
        
        analyzer = BoardGameMechanicsAnalyzer("dummy_api_key", mock_data)
        analyzer.chat_with_gemini = mock_chat_with_gemini  # Replace the actual method with the mock
        
        # Call the method being tested
        analyzer.multiple_games()
        
        # Check that Gemini was not called for the invalid game ("Unknown Game")
        self.assertEqual(mock_chat_with_gemini.call_count, 1)
        
        # Check if the invalid game was handled correctly and printed the appropriate message
        mock_print.assert_any_call("The data you provided does not match the data in the BGG-dataset")
        
        # Ensure that Gemini was called once for the valid game
        self.assertEqual(mock_chat_with_gemini.call_count, 1)

if __name__ == "__main__":
    unittest.main()