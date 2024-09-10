import React, { useState } from 'react';
import './App.css';

function App() {
  // State to hold the data fetched from the FastAPI server
  const [response, setResponse] = useState(null);

  const handleButtonClick = async () => {
      // Fetch the data from the FastAPI backend
      // const res = await fetch('http://localhost:8000/welcome');
      // Parse the response as JSON
      // const data = await res.json();
      // setResponse(data);
      
      fetch('http://localhost:8000/welcome')
      .then(response => {
        if (!response.ok) {
          console.log(response);
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => {
        setResponse(data.message);
      })
      .catch(error => {
        console.error('There has been a problem with your fetch operation:', error);
        setResponse('Failed to fetch data');
      });
  };

  return (
    <div className="App">
      <h1>React Frontend</h1>
      <button onClick={handleButtonClick}>Fetch from Backend</button>
      <p>{response}</p>
    </div>
  );
}

export default App;
