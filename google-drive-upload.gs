/**
 * Google Apps Script for uploading CSV files to Google Drive
 * Deploy this as a web app with execute permissions set to "Anyone"
 */

function doPost(e) {
  try {
    // Parse the request
    const data = JSON.parse(e.postData.contents);
    const csvContent = data.csvContent;
    const fileName = data.fileName;
    const folderId = data.folderId;
    
    // Validate inputs
    if (!csvContent || !fileName || !folderId) {
      return ContentService
        .createTextOutput(JSON.stringify({
          success: false,
          error: 'Missing required parameters'
        }))
        .setMimeType(ContentService.MimeType.JSON);
    }
    
    // Get the target folder
    const folder = DriveApp.getFolderById(folderId);
    
    // Create the CSV file
    const blob = Utilities.newBlob(csvContent, 'text/csv', fileName);
    const file = folder.createFile(blob);
    
    // Log the upload
    console.log(`File uploaded: ${fileName} to folder: ${folder.getName()}`);
    
    return ContentService
      .createTextOutput(JSON.stringify({
        success: true,
        fileId: file.getId(),
        fileName: fileName,
        message: 'File uploaded successfully'
      }))
      .setMimeType(ContentService.MimeType.JSON);
      
  } catch (error) {
    console.error('Error uploading file:', error);
    
    return ContentService
      .createTextOutput(JSON.stringify({
        success: false,
        error: error.toString()
      }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

function doGet(e) {
  // Handle GET requests (for testing)
  return ContentService
    .createTextOutput(JSON.stringify({
      message: 'Google Drive Upload Service is running',
      timestamp: new Date().toISOString()
    }))
    .setMimeType(ContentService.MimeType.JSON);
}

/**
 * Test function to verify the script works
 */
function testUpload() {
  const testData = {
    csvContent: 'Method,Dimension,Statement,Score\nTest,1. Clarity,Test statement,5\n',
    fileName: 'test_upload.csv',
    folderId: '1xPFRZHOywEq4YvdzOBG5WnXlk6ufSYHc'
  };
  
  const mockEvent = {
    postData: {
      contents: JSON.stringify(testData)
    }
  };
  
  const result = doPost(mockEvent);
  console.log('Test result:', result.getContent());
} 