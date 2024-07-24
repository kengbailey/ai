import React, { useState } from 'react';
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
  Link
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';

function App() {
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSearch = async (e) => {
    e.preventDefault();
    setLoading(true);
    setSearchResults(null);
    try {
      const response = await axios.post('http://localhost:8000/search', { query });
      setSearchResults(response.data);
      console.log(response.data);
    } catch (error) {
      console.error('Error performing search:', error);
    }
    setLoading(false);
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
                .filter(result => searchResults.citations.includes(result.index))
                .map(result => (
                  <Grid item xs={12} sm={6} md={4} key={result.index}>
                    <ResultCard result={result} isCited={true} />
                  </Grid>
                ))}
            </Grid>

            <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>Other Results</Typography>
            <Grid container spacing={3}>
              {searchResults.results
                .filter(result => !searchResults.citations.includes(result.index))
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
  );
}

export default App;