import base64
import logging
import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import json
import re
from mimetypes import guess_type
from enum import Enum
import asyncio
import aiohttp

import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from ebaysdk.finding import Connection as eBayAPI
from functions.web_search import text_search
from rich.console import Console
from rich.logging import RichHandler
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt

# Rich setup for better logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger("rich")
console = Console()

class Validation(Enum):
    YES = "YES"
    NO = "NO"

class CardValidation(BaseModel):
    name_correct: Validation
    corrected_name: Optional[str] = None
    hp_correct: Validation
    corrected_hp: Optional[int] = None
    type_correct: Validation
    corrected_type: Optional[str] = None
    card_number_correct: Validation
    corrected_card_number: Optional[str] = None
    rarity_correct: Validation
    corrected_rarity: Optional[str] = None
    description_correct: Validation
    corrected_description: Optional[str] = None

class PokemonCard(BaseModel):
    name: str = Field(..., description="The name of the Pokemon")
    estimated_price: str = Field(..., description="The estimated price of the Pokemon card. Give me your best estimate in USD. From $1 to $2K plus. It's all about priority.")
    hp: int = Field(..., description="The HP value of the Pokemon")
    type: List[str] = Field(..., description="The type(s) of the Pokemon")
    card_number: str = Field(..., description="The card number and set information")
    rarity: str = Field(..., description="The rarity of the card (e.g., Common, Rare, Holo Rare)")
    description: str = Field(..., description="The flavor text or Pokedex entry on the card")

def split_image_into_cards(image: Image.Image, rows=3, cols=3) -> List[Image.Image]:
    """Split a single page image containing a grid of Pokemon cards into individual card images"""
    width, height = image.size
    card_width = width // cols
    card_height = height // rows
    
    cards = []
    for row in range(rows):
        for col in range(cols):
            left = col * card_width
            upper = row * card_height
            right = left + card_width
            lower = upper + card_height
            
            # Crop the individual card
            card = image.crop((left, upper, right, lower))
            
            # Add some padding
            card = add_padding(card, padding=10)
            cards.append(card)
    
    return cards

def add_padding(image: Image.Image, padding: int) -> Image.Image:
    """Add padding around the image"""
    new_width = image.width + 2 * padding
    new_height = image.height + 2 * padding
    padded_image = Image.new(image.mode, (new_width, new_height), (255, 255, 255))
    padded_image.paste(image, (padding, padding))
    return padded_image

def extract_images_from_pdf(pdf_path: str) -> List[str]:
    """Extract images from a PDF file and split them into individual card images"""
    image_paths = []
    try:
        doc = fitz.open(pdf_path)
        os.makedirs('temp_images', exist_ok=True)
        
        for page_index in range(len(doc)):
            page = doc[page_index]
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
            
            # Convert PyMuPDF pixmap to PIL Image
            img_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Split the page image into individual card images
            card_images = split_image_into_cards(img_data)
            
            # Save individual card images
            for card_index, card_image in enumerate(card_images, start=1):
                image_filename = f"temp_images/page{page_index + 1}_card{card_index}.png"
                card_image.save(image_filename, "PNG")
                image_paths.append(image_filename)
                logger.info(f"Extracted card: {image_filename}")
        
        return image_paths
    except Exception as e:
        logger.error(f"Failed to process PDF file: {pdf_path}", exc_info=True)
        raise e

class Client:
    def __init__(self) -> None:
        load_dotenv()
        self._api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self._api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self._azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self._version = os.getenv("AZURE_VERSION")
        self._base_url = f'{self._api_endpoint}/openai/deployments/{self._azure_deployment_name}/chat/completions?api-version={self._version}'

        self.client = AsyncAzureOpenAI(
            api_key=self._api_key,
            api_version=self._version,
            base_url=self._base_url,
        )

    @property
    def azure_deployment_name(self):
        return self._azure_deployment_name

async def validate_card_info(card_info: dict, data_url: str, client: Client) -> Tuple[bool, dict]:
    """Validate the card information using a structured instruction model"""
    try:
        validation_response = await client.client.chat.completions.create(
            model=client.azure_deployment_name,
            messages=[{
                "role": "system",
                "content": """You are a Pokemon card validation expert. Analyze the card and verify each field.
                
                For each field, respond with:
                - YES if the current value is correct
                - NO if the value is incorrect or unclear
                
                If NO, provide the correct value.
                
                IMPORTANT: Respond with ONLY the JSON object, no markdown formatting or other text.
                
                Example response format:
                {
                    "name_correct": "YES",
                    "corrected_name": null,
                    "hp_correct": "NO",
                    "corrected_hp": 90,
                    "type_correct": "NO",
                    "corrected_type": "Fire",
                    "card_number_correct": "YES",
                    "corrected_card_number": null,
                    "rarity_correct": "NO",
                    "corrected_rarity": "Holo Rare",
                    "description_correct": "YES",
                    "corrected_description": null
                }"""
            }, {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"Validate this card analysis:\n{json.dumps(card_info, indent=2)}"
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }]
            }],
            max_tokens=1000,
            temperature=0.1
        )
        
        response_text = validation_response.choices[0].message.content.strip()
        logger.debug(f"Validation response: {response_text}")
        
        try:
            # Clean up the response if it contains markdown formatting
            if response_text.startswith('```'):
                # Remove the opening ```json or ``` and closing ```
                response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
                response_text = re.sub(r'\s*```$', '', response_text)
            
            validation_result = json.loads(response_text)
            validation = CardValidation(**validation_result)
            
            # Build corrections dictionary
            corrections = {}
            if validation.name_correct == Validation.NO and validation.corrected_name:
                corrections['name'] = validation.corrected_name
            if validation.hp_correct == Validation.NO and validation.corrected_hp:
                corrections['hp'] = validation.corrected_hp
            if validation.type_correct == Validation.NO and validation.corrected_type:
                corrections['type'] = validation.corrected_type
            if validation.card_number_correct == Validation.NO and validation.corrected_card_number:
                corrections['card_number'] = validation.corrected_card_number
            if validation.rarity_correct == Validation.NO and validation.corrected_rarity:
                corrections['rarity'] = validation.corrected_rarity
            if validation.description_correct == Validation.NO and validation.corrected_description:
                corrections['description'] = validation.corrected_description
            
            # Determine if manual input is needed
            needs_manual_input = any([
                getattr(validation, f"{field}_correct") == Validation.NO and 
                not getattr(validation, f"corrected_{field}")
                for field in ['name', 'hp', 'type', 'card_number', 'rarity', 'description']
            ])
            
            logger.info(f"Validation complete - Needs manual input: {needs_manual_input}")
            if corrections:
                logger.info(f"Corrections suggested: {corrections}")
            
            return needs_manual_input, corrections
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse validation response: {e}")
            logger.warning(f"Raw response: {response_text}")
            return True, {}
            
    except Exception as e:
        logger.warning(f"Validation failed: {str(e)}")
        logger.debug("Validation error details:", exc_info=True)
        return True, {}

async def analyze_pokemon_card(data_url: str, client: Client, image_path: str = None) -> Dict:
    """Analyze a single Pokemon card image using Azure OpenAI Vision"""
    try:
        # Get basic card info
        basic_info_response = await client.client.chat.completions.create(
            model=client.azure_deployment_name,
            messages=[{
                "role": "system",
                "content": """You are a Pokemon card analyzer. Extract information in this exact format:
                - Name of the Pokemon: <name>
                - HP value: <hp>
                - Type(s): <type>
                - Card number and set: <number>
                - Rarity: <rarity>
                - Description text: <description>"""
            }, {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "Analyze this Pokemon card and extract all relevant information:"
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }]
            }],
            max_tokens=4000,
            temperature=0.6)
        
        card_info = basic_info_response.choices[0].message.content
        
        # Parse the card info into structured data
        parsed_data = {}
        for line in card_info.split('\n'):
            if line.startswith('- '):
                line = line.replace('- ', '')
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    
                    # Special handling for HP value
                    if key == 'hp_value':
                        # Extract just the number from strings like "90 HP"
                        hp_num = ''.join(filter(str.isdigit, value))
                        value = hp_num if hp_num else '0'
                    
                    parsed_data[key] = value

        # Create initial structured data
        structured_data = {
            'name': parsed_data.get('name_of_the_pokemon', 'UNCLEAR'),
            'hp': int(parsed_data.get('hp_value', '0')),
            'type': parsed_data.get('type(s)', 'UNCLEAR'),
            'card_number': parsed_data.get('card_number_and_set', 'UNCLEAR'),
            'rarity': parsed_data.get('rarity', 'UNCLEAR'),
            'description': parsed_data.get('description_text', 'UNCLEAR'),
        }

        # Validate the initial analysis
        needs_manual_input, corrections = await validate_card_info(structured_data, data_url, client)
        
        # Apply any corrections from the validation
        if corrections:
            logger.info("Applying corrections from validation...")
            structured_data.update(corrections)

        # Only use manual input if still needed after validation
        if needs_manual_input and image_path:
            unclear_fields = {
                k: v for k, v in structured_data.items() 
                if v == 'UNCLEAR' or (k == 'hp' and v == 0)
            }
            
            if unclear_fields:
                logger.info("Some fields still unclear after validation. Requesting manual input...")
                manual_info = get_manual_card_info(image_path, unclear_fields)
                structured_data.update(manual_info)

        # Get price estimate using web search
        search_results = None
        urls_json = "[]"
        try:
            search_query = f"{structured_data['name']} pokemon card {structured_data['card_number']} price"
            search_results = text_search(search_query, 5)
            search_results_dict = json.loads(search_results)
            urls = [result['url'] for result in search_results_dict]
            urls_json = json.dumps(urls)
        except Exception as e:
            logger.warning(f"Web search failed: {str(e)}. Proceeding with limited price information.")
            search_results = "No search results available due to rate limiting"

        # Get eBay listings
        ebay_listings = []
        try:
            ebay_api = eBayAPI(config_file='config.yaml', warnings=True)
            
            # Use the verified name for search
            pokemon_name = structured_data['name'].strip()
            card_number = structured_data['card_number'].strip()
            
            if pokemon_name and pokemon_name != 'UNCLEAR':
                # Try multiple search variations
                search_variations = [
                    # If we have a card number, try specific searches first
                    *([] if not card_number or card_number == 'UNCLEAR' else [
                        {'keywords': f"{pokemon_name} {card_number}"},  # Most specific search
                        {'keywords': f"{pokemon_name} pokemon card {card_number}"},  # Full specific search
                    ]),
                    # Then try more general searches
                    {'keywords': f"{pokemon_name} pokemon card"},  # Basic search
                    {'keywords': f"{pokemon_name} card"},  # Name + card
                    {'keywords': pokemon_name}  # Just the name (fallback)
                ]
                
                for search in search_variations:
                    api_request = {
                        **search,  # Add the search keywords
                        'categoryId': '183454',    # Pokemon Card Category
                        'paginationInput': {
                            'entriesPerPage': 10   # Get more results per search
                        },
                        'sortOrder': 'PricePlusShippingLowest',  # Sort by price
                        'outputSelector': ['SellerInfo', 'PictureDetails']
                    }
                    
                    logger.info(f"eBay API Request: {json.dumps(api_request, indent=2)}")
                    response = ebay_api.execute('findItemsAdvanced', api_request)
                    
                    if response.reply.ack == 'Success':
                        logger.info(f"eBay API call successful for '{search['keywords']}'")
                        
                        if hasattr(response.reply, 'searchResult') and hasattr(response.reply.searchResult, 'item'):
                            for item in response.reply.searchResult.item:
                                # Check if this URL is already in our listings
                                url = item.viewItemURL
                                if not any(listing['url'] == url for listing in ebay_listings):
                                    listing = {
                                        'title': item.title,
                                        'price': float(item.sellingStatus.currentPrice.value),
                                        'currency': item.sellingStatus.currentPrice._currencyId,
                                        'url': url,
                                        'search_term': search['keywords']  # Track which search found this
                                    }
                                    ebay_listings.append(listing)
                                    logger.info(f"Found listing: {json.dumps(listing, indent=2)}")
                            
                            # If we found listings with the specific card number, we can stop searching
                            if card_number and card_number != 'UNCLEAR' and any(card_number in listing['title'] for listing in ebay_listings[-10:]):
                                logger.info(f"Found listings with specific card number {card_number}, stopping search")
                                break
                
                logger.info(f"Found total of {len(ebay_listings)} unique eBay listings")
            else:
                logger.warning("No valid Pokemon name found for eBay search")
        except Exception as e:
            logger.warning(f"eBay search failed: {str(e)}")
            logger.warning(f"eBay search error details:", exc_info=True)
        
        # Get price estimate based on all gathered data
        price_prompt = """You are a Pokemon card price analyzer. Analyze the eBay listings and web search results to determine the current market price range.

Rules:
1. Pay special attention to the actual eBay listing prices - these are real market prices
2. If there are multiple eBay listings, use them to establish a price range
3. Consider the card's rarity and condition when shown
4. The price range should reflect the ACTUAL prices seen in the listings
5. If eBay listings show lower prices than web results, favor the eBay prices as they are current market prices
6. Format your response exactly as: "$X.XX - $Y.YY" (e.g., "$0.99 - $15.99")
7. If all listings are the same price, use that as both min and max"""
        
        price_estimate = await client.client.chat.completions.create(
            model=client.azure_deployment_name,
            messages=[{
                "role": "system",
                "content": price_prompt
            }, {
                "role": "user",
                "content": f"""Analyze these data sources and provide a price range that reflects the ACTUAL current market prices:

Card Info:
{card_info}

eBay Listings (CURRENT MARKET PRICES):
{json.dumps(ebay_listings, indent=2)}

Web Search Results (REFERENCE ONLY):
{search_results}

Remember to format your response exactly as "$X.XX - $Y.YY" based primarily on the eBay listing prices."""
            }],
            max_tokens=100,
            temperature=0.3
        )
        
        price_text = price_estimate.choices[0].message.content
        
        # Add the search URLs and eBay listings to the structured data
        structured_data['search_urls'] = urls_json
        structured_data['ebay_listings'] = json.dumps(ebay_listings)
        structured_data['estimated_price'] = price_text
        
        logger.info("Card Analysis Complete")
        logger.debug(f"Structured data: {structured_data}")
        return structured_data
        
    except Exception as e:
        logger.error(f"Failed to analyze Pokemon card: {str(e)}")
        raise

def extract_price_values(price_text):
    """Extract low, middle, and high price values from price text"""
    patterns = [
        r'\$(\d+(?:,\d+)?(?:\.\d+)?)\s*-\s*\$(\d+(?:,\d+)?(?:\.\d+)?)',  # $10-$20
        r'\$(\d+(?:,\d+)?(?:\.\d+)?)\s*to\s*\$(\d+(?:,\d+)?(?:\.\d+)?)',  # $10 to $20
        r'between\s*\$(\d+(?:,\d+)?(?:\.\d+)?)\s*and\s*\$(\d+(?:,\d+)?(?:\.\d+)?)'  # between $10 and $20
    ]
    
    # Try to find a range first
    for pattern in patterns:
        match = re.search(pattern, price_text, re.IGNORECASE)
        if match:
            low = float(match.group(1).replace(',', ''))
            high = float(match.group(2).replace(',', ''))
            middle = (low + high) / 2
            return low, middle, high
    
    # If no range found, look for single price
    single_price = re.search(r'\$(\d+(?:,\d+)?(?:\.\d+)?)', price_text)
    if single_price:
        price = float(single_price.group(1).replace(',', ''))
        return price, price, price
    
    return None, None, None

def make_df(data):
    """Convert structured card data into a DataFrame"""
    try:
        # Create DataFrame from the structured data
        df = pd.DataFrame([data])
        
        # Extract price values
        if 'estimated_price' in df.columns:
            price_low, price_mid, price_high = extract_price_values(df['estimated_price'].iloc[0])
            df['price_float_low'] = price_low
            df['price_float'] = price_mid
            df['price_float_high'] = price_high
        
        # Ensure HP is integer
        df['hp'] = pd.to_numeric(df['hp'], errors='coerce').fillna(0).astype(int)
        
        # Ensure all required columns exist
        required_columns = [
            'name', 'estimated_price', 'price_float_low', 'price_float', 'price_float_high',
            'hp', 'type', 'card_number', 'rarity', 'description', 'search_urls', 'ebay_listings'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = 'UNCLEAR' if col not in ['hp', 'price_float_low', 'price_float', 'price_float_high'] else 0
        
        return df
    except Exception as e:
        logger.error(f"Error creating DataFrame: {str(e)}")
        raise

def init_database():
    """Initialize the database with the required schema"""
    conn = sqlite3.connect('output/pokemon_cards.db')
    cursor = conn.cursor()
    
    # Drop the existing table if it exists
    cursor.execute('DROP TABLE IF EXISTS pokemon_cards')
    
    cursor.execute('''
    CREATE TABLE pokemon_cards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        estimated_price TEXT,
        price_float_low REAL,
        price_float REAL,
        price_float_high REAL,
        hp INTEGER, 
        type TEXT,
        card_number TEXT,
        rarity TEXT,
        description TEXT,
        search_urls TEXT,
        ebay_listings TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    return conn

def save_to_database(df: pd.DataFrame, conn: sqlite3.Connection):
    """Save the analyzed card data to the database"""
    try:
        df['created_at'] = datetime.now()
        df.to_sql('pokemon_cards', conn, if_exists='append', index=False)
        logger.info("Successfully saved card data to SQLite database")
    except Exception as e:
        logger.error("Failed to save data to SQLite database", exc_info=True)
        raise e

def get_manual_card_info(image_path: str, unclear_fields: dict) -> dict:
    """Display the card image and get manual input for unclear fields"""
    # Show the image
    img = Image.open(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    # Get manual input for each unclear field
    manual_info = {}
    print("\nSome fields were unclear. Please provide the following information:")
    
    field_descriptions = {
        'name': 'Pokemon name',
        'hp': 'HP value (number only)',
        'type': 'Type(s) (comma-separated)',
        'card_number': 'Card number and set',
        'rarity': 'Rarity',
        'description': 'Card description'
    }
    
    for field, value in unclear_fields.items():
        if value == 'UNCLEAR' or (field == 'hp' and value == 0):
            while True:
                user_input = input(f"> Enter {field_descriptions.get(field, field)}: ").strip()
                if user_input:  # Ensure input is not empty
                    if field == 'hp':
                        try:
                            manual_info[field] = int(user_input)
                            break
                        except ValueError:
                            print("Please enter a valid number for HP")
                    else:
                        manual_info[field] = user_input
                        break
                print("Input cannot be empty. Please try again.")
    
    return manual_info

async def process_card(image_path: str, client: Client) -> Optional[pd.DataFrame]:
    """Process a single card asynchronously"""
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{base64_image}"
        
        card_data = await analyze_pokemon_card(data_url, client, image_path=image_path)
        df = make_df(card_data)
        print(df)
        return df
    except Exception as e:
        logger.error(f"Failed to process image {image_path}", exc_info=True)
        return None

async def process_pdf(pdf_path: str, client: Client) -> List[pd.DataFrame]:
    """Process a single PDF file asynchronously"""
    try:
        image_paths = extract_images_from_pdf(pdf_path)
        logger.info(f"Extracted {len(image_paths)} cards from {pdf_path}")
        
        # Process all cards concurrently
        tasks = [process_card(image_path, client) for image_path in image_paths]
        results = await asyncio.gather(*tasks)
        
        # Clean up temporary images
        for image_path in image_paths:
            try:
                os.remove(image_path)
            except:
                pass
        
        return [df for df in results if df is not None]
        
    except Exception as e:
        logger.error(f"Failed to process PDF {pdf_path}", exc_info=True)
        return []

async def async_main():
    try:
        input_folder = input("> Enter the path to the folder with Pokemon card PDFs: ").strip()
        client = Client()
        
        # Initialize database connection
        conn = init_database()
        
        filenames = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
        logger.info(f"Found {len(filenames)} PDF file(s) in the input folder.")
        
        if not filenames:
            logger.warning("No PDF files found in the input folder.")
            return
        
        # Process all PDFs concurrently
        pdf_tasks = [
            process_pdf(os.path.join(input_folder, pdf_file), client)
            for pdf_file in filenames
        ]
        
        all_results = await asyncio.gather(*pdf_tasks)
        final_df_list = [df for pdf_results in all_results for df in pdf_results]
        
        if final_df_list:
            try:
                final_df = pd.concat(final_df_list, ignore_index=True)
                
                # Save to Excel
                os.makedirs('output', exist_ok=True)
                today = pd.Timestamp.now().strftime('%Y-%m-%d')
                output_path = f'output/pokemon_cards_{today}.xlsx'
                final_df.to_excel(output_path, index=False)
                logger.info(f"Final output saved to {output_path}")
                
                # Save to SQLite database
                save_to_database(final_df, conn)
                
                print('\n' + '='*100 + f'\nFINAL TABLE PREVIEW:\n{final_df}')
                print(f"\nTotal cards processed: {len(final_df)}")
            except Exception as e:
                logger.error("Failed to save output.", exc_info=True)
        else:
            logger.warning("No cards were successfully processed.")
            
    except Exception as e:
        logger.critical("An unexpected error occurred in the main function.", exc_info=True)
    finally:
        if 'conn' in locals():
            conn.close()

# Modify the entry point
if __name__ == "__main__":
    asyncio.run(async_main()) 