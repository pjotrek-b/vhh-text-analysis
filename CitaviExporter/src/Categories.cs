using System;
using SwissAcademic.Citavi;
using SwissAcademic.Citavi.Shell;

public class Categories
{
	public static void Export(String path)
	{
		Project activeProject = Program.ActiveProjectShell.Project;
		string FOLDERPATH = path+ "\\categories.json";
		try
		{


			System.IO.File.WriteAllBytes(FOLDERPATH, new byte[0]);
			char[] charsToTrim = { ' ', '“', '”', };

			int count = 0;
			using (System.IO.StreamWriter file = new System.IO.StreamWriter(FOLDERPATH, true))
			{
				file.WriteLine("{\r\n\"categories\":[");

				foreach (Category cat in activeProject.AllCategories)
				{

					string parent;
					string parentClassifier;
					if (cat.IsRootCategory)
					{
						parent = "";
						parentClassifier = "";
					}
					else
					{
						parent = cat.ParentCategory.Name.ToString();
						parentClassifier = cat.Parent.Classification.ToString();
					}

					string temp = "";
					if (count > 0)
					{
						temp = ",{\r\n" +
                        "\"Index\":" + cat.Index + ",\r\n" +
						"\"Name\":\"" + cat.Name.Trim(charsToTrim).Replace("\"", "\\\"") + "\",\r\n" +
						"\"Classifier\":\"" + cat.Classification + "\",\r\n" +
						"\"ParentName\":\"" + parent.Trim(charsToTrim).Replace("\"", "\\\"") + "\",\r\n" +
						"\"ParentClassifier\":\"" + parentClassifier + "\",\r\n" +
						"\"Level\":" + cat.Level + "\r\n" +
						"}";
					}
					else
					{
						temp = "{\r\n" +
						"\"Index\":" + cat.Index + ",\r\n" +
						"\"Name\":\"" + cat.Name.Trim(charsToTrim).Replace("\"", "\\\"") + "\",\r\n" +
						"\"Classifier\":\"" + cat.Classification + "\",\r\n" +
						"\"ParentName\":\"" + parent.Trim(charsToTrim).Replace("\"", "\\\"") + "\",\r\n" +
						"\"ParentClassifier\":\"" + parentClassifier + "\",\r\n" +
						"\"Level\":" + cat.Level + "\r\n" +
						"}";
					}




					file.WriteLine(temp);

					count++;

				}

				file.WriteLine("]\r\n}");

				DebugMacro.WriteLine("Success");
			}
		}
		catch (Exception ex)
		{
			DebugMacro.WriteLine("Exception:" + ex.Message);
			DebugMacro.WriteLine(ex.StackTrace);
		}

	}
}
