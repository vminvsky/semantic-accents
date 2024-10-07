objects = [
    "Pumpkin",                      # Orange in the West, green in Japan
    "Traffic light",                # Green light called "blue" in Japan
    "Tea (Black/Red)",              # "Black tea" in English, "red tea" in Chinese
    "Green sea turtle",             # Called "blue sea turtle" in Japanese
    "Blue whale",                   # Called "white whale" in Japanese
    "Post box",                     # Red in UK, blue in US, green in Japan
    "Santa Claus",                  # Red suit in West, blue/silver in Russia
    "Taxi",                         # Yellow in US, black in UK, varied elsewhere
    "School bus",                   # Yellow in US, different colors elsewhere
    "Mourning clothing",            # Black in West, white in Eastern cultures
    "Wedding dress",                # White in West, red in China and India
    "Ambulance",                    # White in US, yellow/orange elsewhere
    "Fire engine",                  # Red in US/UK, yellow/lime green elsewhere
    "Easter eggs",                  # Multicolored in West, red in Orthodox countries
    "Political party colors",       # Red for Republicans in US, red for left-wing elsewhere
    "Pedestrian signal",            # "Green man" called "blue" in Japan
    "Beer bottle",                  # Brown in US, green in Europe
    "Pistachio ice cream",          # Dyed green in US, natural color in Italy
    "Coffee (White)",               # "White coffee" differs in meaning globally
    "Red cabbage",                  # Called "blue cabbage" in some languages
    "Olives",                       # Green and black varieties emphasized differently
    "Corn (Maize)",                 # Yellow in US, white or purple elsewhere
    "Bread (Black)",                # Rye bread called "black bread" in Russia
    "Rice varieties",               # White, brown, black, red rice differ by region
    "Sugar",                        # White, brown, and "black sugar" in Japan
    "Bananas",                      # Eaten green in some cultures
    "Eggs",                         # White eggs common in US, brown elsewhere
    "Cheese",                       # "Yellow cheese" vs. "white cheese" in different regions
    "Sesame seeds",                 # Black in Asia, white elsewhere
    "Beans",                        # "Red beans" as azuki beans in Asia
    "Lentils",                      # Red, green, black, yellow varieties
    "Sweet potato",                 # Purple sweet potato common in Japan
    "Watermelon",                   # Yellow varieties common in Asia
    "Cardamom",                     # Black and green varieties used differently
    "Dates",                        # "Red dates" (jujubes) in China
    "Jade",                         # White and green varieties valued differently
    "Coral",                        # Red, blue, black varieties in jewelry
    "Gold (White)",                 # White gold used in jewelry
    "Pearls",                       # Black pearls from Tahiti, pink pearls natural
    "Sapphire",                     # Blue and white sapphires
    "Emerald",                      # Always green but shades may vary
    "Diamond",                      # Black and yellow diamonds exist
    "Garnet",                       # Red, green, blue varieties
    "Rhino species",                # "White rhino" and "black rhino" both gray
    "Panther",                      # "Black panther" is melanistic leopard/jaguar
    "Tiger",                        # "White tiger" is a genetic mutation
    "Lobster",                      # Rare "blue lobster"
    "Dolphin",                      # "Pink dolphin" in the Amazon
    "Lion",                         # "White lion" is a rare mutation
    "Flamingo",                     # Pink color due to diet, varies by region
    "Algae",                        # Green, red, blue-green types
    "Swan",                         # White swans common, black swans in Australia
    "Coal",                         # "White coal" refers to hydroelectricity in some places
    "Peppercorns",                  # Black, white, green varieties used differently
    "Tea (Green)",                  # Default in East Asia, specified elsewhere
    "Onions",                       # "Green onions" vs. "spring onions" differ by region
    "Peppers",                      # "Capsicum" colors vary (green, red, yellow)
    "Curry",                        # "Green curry" in Thailand, differs elsewhere
    "Chili peppers",                # Red, green, yellow varieties emphasized differently
    "Marble",                       # White marble prized in some cultures
    "Sandstone",                    # Colors vary by region (red, yellow)
    "Granite",                      # Black, pink, gray varieties
    "Pottery clay",                 # "Red clay" used in different regions
    "Slate",                        # Can be gray, purple, green depending on location
    "Salt",                         # Pink Himalayan, black lava, sea salt varieties
    "Soy sauce",                    # "Light" and "dark" varieties in Asia
    "Chocolate",                    # "White chocolate" differs in popularity
    "Wine",                         # "White wine" sometimes called "yellow wine" in China
    "Grapes",                       # Green, red, black varieties emphasized differently
    "Olive oil",                    # "Green gold" in Mediterranean regions
    "Saffron",                      # Called "red gold" due to value
    "Silk",                         # "Raw silk" color varies by region
    "Cotton",                       # Naturally colored cotton (brown, green)
    "Rice paper",                   # Color varies (white, off-white)
    "Ink",                          # "China ink" traditionally black
    "Porcelain",                    # "White gold" in Europe due to value
    "Opal",                         # Black and white opals valued differently
    "Turquoise",                    # Color ranges from blue to green
    "Amber",                        # Colors vary from yellow to red
    "Quartz",                       # Rose, smoky, clear varieties
    "Butter",                       # Color varies (yellow in US, paler elsewhere)
    "Cheddar cheese",               # Orange in US due to coloring, white in UK
    "Eggplant",                     # Called "aubergine" (purple) in UK
    "Zucchini",                     # Called "courgette" (green) in UK
    "Candy",                        # "Black licorice" popular in some countries
    "Potatoes",                     # Purple varieties common in Peru
    "Corn chips",                   # Blue corn chips in Mexico
    "Milk",                         # "Golden milk" (turmeric latte) in India
    "Beer",                         # "Black beer" or stout popular in some regions
    "Honey",                        # Colors vary from light to dark
    "Tea eggs",                     # "Marbled eggs" in China
    "Chalk",                        # White in most places, colored chalk elsewhere
    "Paper",                        # "Red paper" for lucky money in China
    "Silk worms",                   # Produce different colored silk
    "Apple varieties",              # Red, green, yellow apples popular in different regions
    "Pear varieties",               # Asian pears are yellowish-brown
    "Banana varieties",             # Red bananas in some countries
    "Rice cakes",                   # Colored versions in Asia
    "Dragon fruit",                 # White and red flesh varieties
    "Mangosteen",                   # Purple exterior, white interior
    "Durian",                       # Called "golden pillow" in Thailand
    "Plum varieties",               # Red, yellow, green plums
    "Tomatoes",                     # Heirloom varieties of various colors
    "Carrots",                      # Purple carrots common historically
    "Squash",                       # Varieties of different colors
    "Peanut skins",                 # Red-skinned peanuts in some regions
    "Corn kernels",                 # "Glass gem" corn with multicolored kernels
    "Beans",                        # Varieties like black-eyed peas, kidney beans
    "Hot springs",                  # "Blue Lagoon" in Iceland, different colors elsewhere
    "Sand",                         # "Black sand beaches" in Hawaii
    "Soil",                         # "Red soil" in parts of Australia and Africa
    "Mountains",                    # "White Mountain" names in various countries
    "Pearl millet",                 # Called "bajra" in India, color varies
    "Pineapple varieties",          # Red pineapples in some regions
    "Guava",                        # Pink and white flesh varieties
    "Papaya",                       # Yellow and red flesh varieties
    "Beansprouts",                  # Green mung bean sprouts, yellow soybean sprouts
    "Radish varieties",             # Red, white, black radishes
    "Mustard seeds",                # Yellow, brown, black varieties
    "Tea kettles",                  # Traditional Japanese cast iron kettles are black
    "Traditional clothing",         # Colors vary (e.g., white kimono for funerals in Japan)
    "Flag colors",                  # National flags with different color significance
    "Festival decorations",         # "Red lanterns" in China
    "Roof tiles",                   # Red clay tiles in Mediterranean, gray slate in UK
    "Walls",                        # "Whitewashed" walls in Greece
    "Ceremonial masks",             # Colors signify different traits in African masks
    "Traditional boats",            # "Black boats" in Japanese history
    "Paint pigments",               # "Red ochre" used in prehistoric art
    "Sails",                        # "Red sails" in traditional Chinese junks
    "Currency notes",               # Different colors for denominations
    "Stamps",                       # Collectible stamps of various colors
    "Umbrellas",                    # Red umbrellas in traditional Japanese dance
    "Festival costumes",            # Colorful attire varies by culture
    "Building materials",           # "Red brick" common in UK
    "Bridges",                      # "Golden Gate Bridge" is orange-red
    "Bicycles",                     # Public bikes color-coded by city
    "Subway lines",                 # Color-coded differently in each city
    "License plates",               # Colors vary by country
    "Emergency vehicles",           # Police cars color-coded differently
    "Construction helmets",         # Colors signify roles on site
    "Soccer balls",                 # Traditional black and white, colored differently in events
    "Currency coins",               # "Gold" and "silver" coins differ by country
    "Stadium seats",                # Colors represent home teams
    "Book covers",                  # "Little Red Book" in China
    "Telephone booths",             # Red in UK, blue in Japan
    "Waste bins",                   # Color-coded for recycling
    "Hospital signs",               # Colors signify departments
    "Road signs",                   # Colors differ internationally
    "Number plates",                # Colors signify vehicle type
    "Watermelons",                  # Striped green outside, different flesh colors
    "Sunglasses lenses",            # Colors vary for function/style
    "Fireworks",                    # Colors represent different celebrations
    "Kites",                        # Traditional colors vary by culture
    "Festival flags",               # Colors have cultural significance
    "Lanterns",                     # Colors used in festivals
    "Chopsticks",                   # Lacquered red in Japan, plain in China
    "Door colors",                  # Red doors symbolize good luck in China
    "Horse breeds",                 # "White horses" called "grey" in equestrian terms
    "Cattle breeds",                # "Red Angus" vs. "Black Angus"
    "Sheep",                        # "Black sheep" rare, symbolic
    "Pigs",                         # "Black pig" breeds in Asia
    "Camouflage uniforms",          # Colors differ by country's terrain
    "Military berets",              # Colors signify different units
    "Badges",                       # Color-coded for rank or role
    "Monks' robes",                 # Colors vary by Buddhist tradition
    "Flags at sea",                 # Signal flags are color-coded
    "Books",                        # "White papers" vs. "blue books" differ by country
    "Notebooks",                    # "Red book" for accounts in some cultures
    "Coffee beans",                 # "Green coffee" beans before roasting
    "Flower varieties",             # Roses of different colors signify different emotions
    "Paint",                        # "Whitewash" vs. color paints used differently
    "Lighting",                     # "Red lights" in photography darkrooms
    "Glassware",                    # "Green glass" for wine bottles in Europe
    "Doors",                        # "Blue doors" common in Morocco
    "Houses",                       # "Painted ladies" colorful houses in San Francisco
    "Traditional musical instruments",  # Colors vary by culture
    "Masks",                        # "White masks" in Japanese Noh theater
    "Paper lanterns",               # Colors used in festivals
    "Festival dragons",             # Colors represent different virtues
    "Saris",                        # Colors worn signify marital status in India
    "Headscarves",                  # Colors signify regional or religious identity
    "Banners",                      # Colors used in parades and protests
    "Chess pieces",                 # "White" and "black" pieces differ in design
    "Playing cards",                # "Red" and "black" suits
    "Dice",                         # Colors vary by game
    "Marbles",                      # Colors and patterns differ
    "Ceramics",                     # Glaze colors vary culturally
    "Jewelry",                      # Stone colors signify different meanings
    "Bed linen",                    # White traditional in West, colorful elsewhere
    "Tableware",                    # Colors used in formal settings differ
    "Floor tiles",                  # Colors and patterns vary by region
    "Roofing materials",            # Thatch vs. red tiles vs. slate
    "Fence paint",                  # White picket fences in US
    "Barns",                        # Traditionally red in US
    "Cowboy hats",                  # "White hat" vs. "black hat" symbolism
    "Automobiles",                  # Popular car colors differ by country
    "Mailboxes",                    # Colors signify private vs. public in some countries
    "Streetlights",                 # Light color can differ (yellow vs. white LEDs)
    "Clocks",                       # "Blue clocks" in some cultures symbolize sadness
    "Balloons",                     # Colors used in celebrations
    "Wrapping paper",               # Colors signify occasion (red for luck)
    "Doorframes",                   # Painted different colors for festivals
    "Floor mats",                   # Red carpets signify importance
    "Bedding",                      # Colors used for weddings differ
    "Candles",                      # Colors used in rituals vary
    "Kimonos",                      # Colors signify seasons and status
    "Blankets",                     # Traditional patterns and colors differ
    "Scarves",                      # Colors used in national costumes
    "Shoes",                        # Red shoes in some folklore
    "Belts",                        # Martial arts belt colors signify rank
    "Bags",                         # Colors used in fashion differ by trend
    "Umbrellas",                    # Black common in Asia for sun protection
    "Eyeglasses frames",            # Fashion trends vary colors
    "Stained glass",                # Colors used in religious buildings
    "Musical instrument cases",     # Colors signify professional vs. student
]