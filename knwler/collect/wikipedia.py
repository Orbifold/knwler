import wikipediaapi


import random
from knwler.cache import cache_wikipedia_response, get_cached_wikipedia_response
# add your own preferred articles here
library = {
    "mathematics": [
        "Isaac Newton",
        "Leonhard Euler",
        "Carl Friedrich Gauss",
        "Emmy Noether",
        "Srinivasa Ramanujan",
        "Euclid",
        "Archimedes",
        "Ada Lovelace",
        "Alan Turing",
        "John von Neumann",
        "Évariste Galois",
        "Sophie Germain",
        "David Hilbert",
        "Kurt Gödel",
        "Maryam Mirzakhani",
        "Paul Erdős",
        "Bernhard Riemann",
        "Pierre de Fermat",
        "Blaise Pascal",
        "René Descartes",
        "Pythagoras",
        "Hypatia",
        "Roger Penrose",
        "Omar Khayyam",
        "Fibonacci",
    ],
    "biology": [
        "Charles Darwin",
        "Gregor Mendel",
        "Louis Pasteur",
        "Jane Goodall",
        "Rachel Carson",
        "Rosalind Franklin",
        "Barbara McClintock",
        "Carl Linnaeus",
        "James Watson",
        "Francis Crick",
        "Richard Dawkins",
        "E. O. Wilson",
        "Lynn Margulis",
        "Alexander Fleming",
        "Jonas Salk",
        "Francis Collins",
    ],
    "business": [
        "Steve Jobs",
        "Bill Gates",
        "Warren Buffett",
        "Jeff Bezos",
        "Elon Musk",
        "Henry Ford",
        "Andrew Carnegie",
        "John D. Rockefeller",
        "Peter Drucker",
        "Jack Welch",
        "Mary Barra",
        "Indra Nooyi",
        "Satya Nadella",
        "Tim Cook",
        "Mark Zuckerberg",
    ],
    "finance": [
        "Adam Smith",
        "John Maynard Keynes",
        "Milton Friedman",
        "Benjamin Graham",
        "Ray Dalio",
        "George Soros",
        "Janet Yellen",
        "Paul Volcker",
        "Alan Greenspan",
        "Christine Lagarde",
        "Burton Malkiel",
        "Eugene Fama",
        "Robert Shiller",
        "Hyman Minsky",
        "Friedrich Hayek",
    ],
    "arts": [
        "Leonardo da Vinci",
        "Pablo Picasso",
        "Vincent van Gogh",
        "Frida Kahlo",
        "Michelangelo",
        "Claude Monet",
        "Georgia O'Keeffe",
        "Andy Warhol",
        "Salvador Dalí",
        "Rembrandt",
        "Yayoi Kusama",
        "Banksy",
        "Jean-Michel Basquiat",
        "Gustav Klimt",
        "Henri Matisse",
    ],
    "literature": [
        "William Shakespeare",
        "Jane Austen",
        "Leo Tolstoy",
        "Fyodor Dostoevsky",
        "Virginia Woolf",
        "Gabriel García Márquez",
        "Toni Morrison",
        "Ernest Hemingway",
        "Maya Angelou",
        "Charles Dickens",
        "Emily Dickinson",
        "James Joyce",
        "Chinua Achebe",
        "Haruki Murakami",
        "Margaret Atwood",
    ],
}


class WikipediaCollector:
    """
    Collector for fetching articles from Wikipedia.
    """

    @staticmethod
    async def fetch_article(page_title: str, no_cache: bool = False) -> dict:
        """
        Fetches the text of a Wikipedia page given its title.
        """
        cached = get_cached_wikipedia_response(page_title)
        if cached is not None and not no_cache:
            cached["cached"] = True
            return cached

        wiki_wiki = wikipediaapi.Wikipedia(
            user_agent="Knwl (https://knwler.com)",
            language="en",
            extract_format=wikipediaapi.ExtractFormat.WIKI,
        )
        page = wiki_wiki.page(page_title)
        if page.exists():
            doc= {
                "text": page.text,
                "name": page.title,
                "id": page.fullurl,
                "description": page.summary,
                "content": page.text,
                "cached": False,
            }
            cache_wikipedia_response(page_title, doc)
            return doc
        return None

    @staticmethod
    async def get_random_library_article(category: str = None) -> str:
        """
        Fetches a random article from the specified category in the local library cache or from Wikipedia if not cached.
        If no category is specified, a random category is chosen.
        """
        if category is None:
            category = random.choice(list(library.keys()))
        if category not in library:
            raise ValueError(f"Category '{category}' not found in library.")

        article = random.choice(library[category])
        return await WikipediaCollector.fetch_article(article)

    @staticmethod
    def take_random_wiki_title() -> str:
        """
        Returns a random Wikipedia article title from the library.
        """
        category = random.choice(list(library.keys()))
        title = random.choice(library[category])
        return title
