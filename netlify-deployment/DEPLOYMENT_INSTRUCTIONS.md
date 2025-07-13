# Netlify Deployment Instructions

## Overview
This research study site has been configured to use Netlify Forms for collecting study results. The site will submit evaluation data directly to Netlify instead of downloading JSON files.

## Features Configured
- ✅ Netlify Forms integration for both study types (HVA-Highlights and HVA-GradCAM)
- ✅ AJAX form submission with proper error handling
- ✅ Bilingual support (English and Chinese)
- ✅ Structured data collection with individual evaluation dimensions
- ✅ Security headers and CSP configuration
- ✅ Custom 404 handling

## Deployment Steps

### 1. Deploy to Netlify
1. **Via Git Integration (Recommended)**:
   - Connect your repository to Netlify
   - Set build directory to `netlify-deployment`
   - Netlify will automatically detect the `netlify.toml` configuration

2. **Via Drag & Drop**:
   - Zip the entire `netlify-deployment` folder
   - Drag and drop to Netlify dashboard
   - The site will be deployed automatically

### 2. Enable Netlify Forms
**IMPORTANT**: After deployment, you must enable form detection:

1. Go to your Netlify site dashboard
2. Navigate to **Forms** in the left sidebar
3. Click **Enable form detection** 
4. Deploy your site again (this triggers form detection)

### 3. Verify Form Setup
After enabling forms and redeploying:
1. Go to **Forms** in your Netlify dashboard
2. You should see a form named **"study-results"**
3. The form should show all the configured fields

## Form Data Structure
The form collects the following data:
- **study-type**: "hva-highlights" or "hva-gradcam"
- **first-method** / **second-method**: Method names (HVA-X, Highlights, Grad-CAM)
- **language**: "en" or "zh"
- **order**: Randomized order ("hva-first" or other)
- **timestamp**: Submission timestamp
- **results-json**: Complete JSON results for backup
- **Evaluation scores**: Individual scores for each dimension:
  - first-clarity, first-understandable, first-completeness, etc.
  - second-clarity, second-understandable, second-completeness, etc.

## Form Notifications (Optional)
To receive email notifications for form submissions:
1. Go to **Forms** > **Settings** > **Notifications**
2. Click **Add notification**
3. Choose **Email notification**
4. Enter your email address
5. Save the notification

## Testing the Forms
1. Complete a study session
2. Submit the final evaluation
3. Check **Forms** > **study-results** in your Netlify dashboard
4. Verify the submission contains all expected data

## Data Export
To export form submissions:
1. Go to **Forms** > **study-results**
2. Click **Export CSV** or use the Netlify API
3. All submissions will be exported with timestamps

## Security Notes
- Form submissions are protected by Netlify's built-in spam protection
- CSP headers are configured to prevent XSS attacks
- All form data is stored securely in Netlify's infrastructure

## Troubleshooting

### Forms Not Appearing
- Ensure form detection is enabled in Netlify dashboard
- Redeploy the site after enabling form detection
- Check that HTML forms have `data-netlify="true"` attribute

### Form Submission Fails
- Check browser console for JavaScript errors
- Verify form field names match HTML form inputs
- Ensure proper Content-Type headers in fetch request

### Missing Data
- Confirm all form fields are populated before submission
- Check that dimension names match between JS and HTML
- Verify timestamp generation works correctly

## Support
If you encounter issues:
1. Check Netlify's form documentation
2. Review browser console for errors
3. Verify all form fields are properly configured
4. Test form submission in different browsers

## File Structure
```
netlify-deployment/
├── index.html                          # Main landing page
├── hva-highlights-comparison.html      # HVA-X vs Highlights study
├── hva-gradcam-comparison.html         # HVA-X vs Grad-CAM study
├── video-test.html                     # Video testing page
├── videos/                             # Video assets
├── netlify.toml                        # Netlify configuration
├── _redirects                          # URL redirects
└── DEPLOYMENT_INSTRUCTIONS.md          # This file
```

The site is now ready for deployment with full Netlify Forms integration! 