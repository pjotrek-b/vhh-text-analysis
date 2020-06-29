using System;
using SwissAcademic.Citavi;
using SwissAcademic.Citavi.Shell;

public class Keywords
{
	public static void Export(String path)
	{
		Project activeProject = Program.ActiveProjectShell.Project;
		string FOLDERPATH = path+ "\\keywords.json";
		try
		{
		
			System.IO.File.WriteAllBytes(FOLDERPATH, new byte[0]);
			char[] charsToTrim = { ' ', '“', '”', };
			int count = 0;

			using (System.IO.StreamWriter file = new System.IO.StreamWriter(FOLDERPATH, true))
			{
				file.WriteLine("{\r\n\"keywords\":[");


				foreach (Keyword key in activeProject.Keywords)
				{
					string temp = "";
					if (count > 0)
					{
						temp = ",\r\n" +
						"{\r\n" +
						"\"Keyword\":\"" + key.FullName.Trim(charsToTrim).Replace("\"", "\\\"") + "\"\r\n" +
                        ", \"Notes\":\"" + key.Notes.Trim(charsToTrim).Replace("\"", "\\\"") + "\"\r\n" +
						"}";
					}
					else
					{
						temp = "{\r\n" +
						"\"Keyword\":\"" + key.FullName.Trim(charsToTrim).Replace("\"", "\\\"") + "\"\r\n" +
						", \"Notes\":\"" + key.Notes.Trim(charsToTrim).Replace("\"", "\\\"") + "\"\r\n" +
						"}";
                    }




					file.WriteLine(temp);


					count++;

				}


				file.WriteLine("]\r\n}");

			}
			DebugMacro.WriteLine("Success");
		}
		catch (Exception e)
		{
			DebugMacro.WriteLine("Exception:" + e.Message);
			DebugMacro.WriteLine(e.StackTrace);
		}
	}
}
