import re

import pandas as pd
from unidecode import unidecode


def extract_link(s):
    # Regex pattern to find <a> tags and extract href
    pattern = r'<a[^>]+href="([^"]*)"[^>]*>.*?</a>'
    matches = re.findall(pattern, s)

    # If matches found, replace the entire <a> tag with the link
    if matches:
        for match in matches:
            s = s.replace('<a href="' + match + '">', match)
            s = s.replace("</a>", "")
            # replace other start <a> tags with href
            s = re.sub('<a[^>]+href="' + re.escape(match) + '"[^>]*>', match + " ", s)
    return s


def extract_italic_content(s):
    # Regex pattern to find <i> tags and extract content
    pattern = r"<i[^>]*>([^<]*)</i>"
    matches = re.findall(pattern, s)

    # If matches found, replace the entire <i> tag with the content
    if matches:
        for match in matches:
            s = re.sub(r"<i[^>]*>" + re.escape(match) + "</i>", match, s)
    return s


# Load the data from your CSV file
df = pd.read_csv("/Orchestrator/facebook_politcal_ads_kaggle.csv")

# Apply the unidecode function to each message to replace any non-ASCII characters
df["message"] = df["message"].apply(lambda x: unidecode(x))

# Keep only the 'message' column
# df = df[['message']]

# Remove "<p>", "</p>", "<p>" and "</p>"
df["message"] = df["message"].str.replace("<p>", "", regex=False)
df["message"] = df["message"].str.replace("</p>", "", regex=False)
df["message"] = df["message"].str.replace('"<p>', "", regex=False)
df["message"] = df["message"].str.replace('</p>"', "", regex=False)

# Remove any sequences starting with "<span" and ending with "</span>", including contents
df["message"] = df["message"].apply(lambda x: re.sub("<span.*?>.*?</span>", "", x))
# Additional line to remove any "</span>" instances
df["message"] = df["message"].str.replace("</span>", "", regex=False)

# Remove <a> tags, keep links only
df["message"] = df["message"].apply(extract_link)

# Remove any sequences starting with "<div" and ending with ">", keeping the content
df["message"] = df["message"].apply(lambda x: re.sub("<div[^>]*>", "", x))

# Remove any "</div>" instances
df["message"] = df["message"].str.replace("</div>", "", regex=False)

# Remove any sequences starting with "<i" and ending with "</i>", including contents
df["message"] = df["message"].apply(lambda x: re.sub("<i.*?>.*?</i>", "", x))

# Remove any sequences starting with "<a" and ending with "</a>", including contents
df["message"] = df["message"].apply(lambda x: re.sub("<a.*?>.*?</a>", "", x))

# Remove any empty "<a>" tags with attributes but without content
df["message"] = df["message"].apply(lambda x: re.sub("<a[^>]*></a>", "", x))

# Remove "<br>" tag
df["message"] = df["message"].str.replace("<br>", "", regex=False)

# Remove "<ul>" and "</ul>" tags, keep content only
df["message"] = df["message"].apply(lambda x: re.sub("<ul[^>]*>", "", x))
df["message"] = df["message"].str.replace("</ul>", "", regex=False)

# Remove "<li>" and "</li>" tags, keep content only
df["message"] = df["message"].apply(lambda x: re.sub("<li[^>]*>", "", x))
df["message"] = df["message"].str.replace("</li>", "", regex=False)

# After all your transformations, remove rows where 'message' is null or empty
df["message"].replace("", pd.NA, inplace=True)  # Replace empty strings with NA
df.dropna(subset=["message"], inplace=True)  # Drop rows where 'message' is NA

# Rename 'message' column to 'text'
df.rename(columns={"message": "text"}, inplace=True)

# Save the result back to a CSV file
df.to_csv("facebook_politcal_ads_clean.csv", index=False)
