import React, { useState } from 'react';
import './App.css';

function App() {

  // State to hold the response from the welcome route
  const [response, setResponse] = useState('');

  // State to hold the clothing type value
  const [selectedCloth, setSelectedCloth] = useState('');

  // State to hold the image received from the generate route
  const [imageSrc, setImageSrc] = useState('');

  // Test the connection to the FastAPI server
  const handleButtonClick = async () => {      
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

  // Handle the change event for clothing type
  const handleSelectChange = async (event) => {
    setSelectedCloth(event.target.value);
  };

  const handleGenImage = async () => {     
    const url = `http://localhost:8000/generate`; 
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ "idx": selectedCloth }),
    })

    if (response.ok) {
      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      setImageSrc(imageUrl);
    } else {
      console.error('Failed to generate image');
    }

};


  return (  
    <div className="App">
      <h1>Generate images similar to<br />FashionMNIST dataset!</h1>
      <button onClick={handleButtonClick}>Check that Backend is responsive</button>
      <p>{response}</p>
      <br />
      <label className='field'>
        Select clothing type : 
        <select value={selectedCloth} onChange={handleSelectChange} className='dropdown'>
          <option value=""></option>
          <option value="1">Pants</option>
          <option value="2">Sweat-shirt</option>
          <option value="3">Dress</option>
          <option value="4">Sweater</option>
          <option value="5">Sandal</option>
          <option value="6">T-shirt</option>
          <option value="7">Shoe</option>
          <option value="8">Bag</option>
          <option value="9">Boot</option>
        </select>
      </label>
      <button onClick={handleGenImage}>Generate image!</button>
      <br />
      <br />
      <div>
        {imageSrc ? (
          <img
          src={imageSrc}
          alt="Fetched from server"
          style={{ width: '300px', height: '300px' }} // Adjust the width and height as needed
          />
        ) : (
        <p>Loading image...</p>
        )}
      </div>
    </div>
  );
}

export default App;
