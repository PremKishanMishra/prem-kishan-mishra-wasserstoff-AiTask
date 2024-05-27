from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
from app.config import API_URL

def fetch_posts():
    transport = RequestsHTTPTransport(url=API_URL, verify=True, retries=3)
    client = Client(transport=transport, fetch_schema_from_transport=True)

    query = gql("""
    {
      posts {
        nodes {
          id
          content
        }
      }
    }
    """)
    
    response = client.execute(query)
    return response
