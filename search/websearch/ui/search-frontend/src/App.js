import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  TextField,
  Button,
  Card,
  CardContent,
  CardActions,
  Grid,
  Chip,
  Box,
  Skeleton,
  Fade,
  Link,
  ThemeProvider,
  createTheme,
  CssBaseline
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';

const refinedMochaTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#A67C52', // Rich coffee brown
    },
    secondary: {
      main: '#D4B08C', // Creamy latte
    },
    background: {
      default: '#1E1512', // Very dark brown, almost black
      paper: '#2C2119', // Dark roast coffee
    },
    text: {
      primary: '#E6D9CC', // Light cream
      secondary: '#B3A59A', // Soft mocha
    },
    error: {
      main: '#A85751', // Reddish brown
    },
    warning: {
      main: '#D6AD60', // Warm caramel
    },
    info: {
      main: '#7D9BA1', // Cool blue-grey
    },
    success: {
      main: '#8DA47E', // Muted green
    },
  },
  typography: {
    fontFamily: "'Raleway', 'Roboto', 'Helvetica', 'Arial', sans-serif",
    h1: {
      fontWeight: 300,
    },
    h2: {
      fontWeight: 300,
    },
    h3: {
      fontWeight: 400,
    },
    h4: {
      fontWeight: 400,
    },
    h5: {
      fontWeight: 500,
    },
    h6: {
      fontWeight: 500,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 8px 16px 0 rgba(0,0,0,0.2)',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          background: 'linear-gradient(45deg, #2C2119 30%, #3C2E24 90%)',
          boxShadow: '0 3px 5px 2px rgba(60, 46, 36, .3)',
        },
      },
    },
  },
});

const mochaTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#cb6077',
    },
    secondary: {
      main: '#beb55b',
    },
    background: {
      default: '#3b3228',
      paper: '#534636',
    },
    text: {
      primary: '#d0c8c6',
      secondary: '#a89bb9',
    },
  },
});

function App() {
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [searchStatus, setSearchStatus] = useState('');

  // const handleSearch = async (e) => {
  //   e.preventDefault();
  //   setLoading(true);
  //   setSearchResults(null);
  //   try {
  //     const response = await axios.post('http://localhost:8000/search', { query });
  //     setSearchResults(response.data);
  //     console.log(response.data);
  //   } catch (error) {
  //     console.error('Error performing search:', error);
  //   }
  //   setLoading(false);
  // };

  const handleSearch = async (e) => {
    e.preventDefault();
    setLoading(true);
    setSearchResults(null);
    setSearchStatus('');

    const eventSource = new EventSource(`http://localhost:8000/search-sse?query=${encodeURIComponent(query)}`);

    eventSource.onmessage = (event) => {
      setSearchStatus(event.data);
    };

    eventSource.addEventListener('result', (event) => {
      const result = JSON.parse(event.data);
      setSearchResults(result);
      setLoading(false);
      eventSource.close();
    });

    eventSource.onerror = (error) => {
      console.error('Error:', error);
      setLoading(false);
      eventSource.close();
    };
  };

  const ResultCard = ({ result, isCited }) => (
    <Fade in={true} timeout={500}>
      <Card elevation={3} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <CardContent sx={{ flexGrow: 1 }}>
          <Typography variant="h6" component="div" gutterBottom>
            {result.title}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {result.snippet}
          </Typography>
        </CardContent>
        <CardActions>
          <Button 
            size="small" 
            endIcon={<OpenInNewIcon />}
            href={result.url}
            target="_blank"
            rel="noopener noreferrer"
          >
            Read More
          </Button>
          {isCited && <Chip label="Cited" color="primary" size="small" />}
        </CardActions>
      </Card>
    </Fade>
  );

  return (
    <div>
      <ThemeProvider theme={refinedMochaTheme}>
        <CssBaseline />
        <div>
          <AppBar position="static">
            <Toolbar>
              <Typography variant="h6">Search Application</Typography>
            </Toolbar>
          </AppBar>
          <Container maxWidth="md" sx={{ mt: 4 }}>
            <form onSubmit={handleSearch}>
              <Grid container spacing={2} alignItems="center">
                <Grid item xs>
                  <TextField
                    fullWidth
                    variant="outlined"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Enter your search query"
                  />
                </Grid>
                <Grid item>
                  <Button
                    variant="contained"
                    type="submit"
                    disabled={loading}
                    startIcon={<SearchIcon />}
                  >
                    {loading ? 'Searching...' : 'Search'}
                  </Button>
                </Grid>
              </Grid>
            </form>

            {loading && (
              <Box sx={{ mt: 4 }}>
                <Skeleton variant="rectangular" height={100} />
                <Box sx={{ mt: 2 }}>
                  <Skeleton />
                  <Skeleton width="60%" />
                </Box>
              </Box>
            )}

            {searchStatus && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="body1">{searchStatus}</Typography>
              </Box>
            )}

            {searchResults && (
              <Box sx={{ mt: 4 }}>
                <Typography variant="h5" gutterBottom>Search Results</Typography>
                <Card sx={{ mb: 2 }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>Answer</Typography>
                    <Typography variant="body1">{searchResults.answer}</Typography>
                  </CardContent>
                </Card>

                <Typography variant="h6" gutterBottom>Relevant Results</Typography>
                <Grid container spacing={3}>
                  {searchResults.results
                    .filter(result => result.is_cited === true)
                    .map(result => (
                      <Grid item xs={12} sm={6} md={4} key={result.index}>
                        <ResultCard result={result} isCited={true} />
                      </Grid>
                    ))}
                </Grid>

                <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>Other Results</Typography>
                <Grid container spacing={3}>
                  {searchResults.results
                    .filter(result => result.is_cited === false)
                    .map(result => (
                      <Grid item xs={12} sm={6} md={4} key={result.index}>
                        <ResultCard result={result} isCited={false} />
                      </Grid>
                    ))}
                </Grid>
              </Box>
            )}
          </Container>
        </div>
      </ThemeProvider>
      
    </div>
  );
}

export default App;