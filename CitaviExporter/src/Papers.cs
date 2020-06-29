using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using SwissAcademic.Citavi;
using SwissAcademic.Citavi.Shell;

public class Papers
{
	public static void Export(String path)
	{
		Project activeProject = Program.ActiveProjectShell.Project;
		string FOLDERPATH = path+ "\\references.json";
		try
		{
			
			System.IO.File.WriteAllBytes(FOLDERPATH, new byte[0]);

			char[] charsToTrim = { ' ', '“', '”', };
			int count = 0;
			using (System.IO.StreamWriter file = new System.IO.StreamWriter(FOLDERPATH, true))
			{
				file.WriteLine("{\r\n\"references\":[");

			    var customFields =
			        activeProject.ProjectSettings.CustomFields.ToDictionary((setting) => (setting.Property.PropertyId.ToString()));
				
				foreach (Reference reference in activeProject.References)
				{
					
					string temp = "";
					string allProps = "";
				    
					PropertyInfo[] properties = reference.GetType().GetProperties();
					
					foreach (var prop in properties)
					{
					    var valueIsText = true;
                        string key = prop.Name;
					    string value = prop.GetValue(reference).ToStringSafe();

                        if (prop.Name.StartsWith("CustomField"))
					    {
					        key = customFields[prop.Name].LabelText;
					    }
                        else if (prop.Name == "Groups")
                        {
                            valueIsText = false;
                            var v = (ReferenceGroupCollection) prop.GetValue(reference);
                            if (v.Count == 0) value = "[]";
                            else value = "[" + v.Select(c => "\"" + c.FullName + "\"").Aggregate((i, j) => i + "," + j) + "]";
                        }
                        else if (prop.Name == "Keywords")
					    {
                            valueIsText = false;
					        var v = (ReferenceKeywordCollection)prop.GetValue(reference);
					        if (v.Count == 0) value = "[]";
					        else value = "[" + v.Select(c => "\"" + c.Name + "\"").Aggregate((i, j) => i + "," + j) + "]";
					    }
                        else if (prop.Name == "Categories")
                        {
                            valueIsText = false;
                            var v = (ReferenceCategoryCollection)prop.GetValue(reference);
                            if (v.Count == 0) value = "[]";
                            else value = "[" + v.Select(c => "\"" + c.Classification + "\"").Aggregate((i, j) => i + "," + j) + "]";
                        }
                        else if (prop.Name == "Abstract")
					    {
                            value = reference.Abstract.Text;
                        }
                        else if (prop.Name == "EntityLinks")
                        {
                            valueIsText = false;
                            var v = ((IEnumerable<EntityLink>) prop.GetValue(reference)).ToList();
                            if (v.Count == 0) value = "[]";
                            else value = "[" + v.Select(i => $"{{\"Id\":\"{i.Id.ToString()}\",\"Text\":\"{i.ToString()}\"}}")
                                    .Aggregate((i, j) =>  i + "," + j) + "]";
                        }
                        else if (prop.Name == "Locations")
                        {
                            valueIsText = false;
                            var v = ((IEnumerable<Location>)prop.GetValue(reference)).ToList();
                            if (v.Count == 0) value = "[]";
                            else value = "[" + v.Select(i => $"{{\"Id\":\"{i.Id.ToString()}\",\"FullName\":\"{i.FullName}\",\"Address\":\"{i.Address}\"}}")
                                                .Aggregate((i, j) => i + "," + j) + "]";
                        }

					    if (valueIsText) value = value.Trim(charsToTrim).Replace("\"", "\\\"");

                        if (!prop.Name.Equals("FormattedReference") 
                            && !prop.Name.Equals("IsPropertyChangeNotificationSuspended") 
                            && !prop.Name.Equals("StaticIds")
                            && !prop.Name.Equals("TableOfContents"))
						{
							if (!allProps.Equals(""))
							{
								allProps += ",\r\n";
							}
                            
							if (valueIsText) allProps += "\""+key+"\":\"" +value+ "\"";
                            else allProps += "\"" + key + "\":" + value + "";
                        }

					}

					if (count > 0)
					{
						temp = ",{\r\n" +
						allProps + "\r\n"+
						"}";
					}
					else

					{
						temp = "{\r\n" +
						allProps + "\r\n" +
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
		    System.Windows.Forms.MessageBox.Show("Exception:" + e.Message);
		    System.Windows.Forms.MessageBox.Show(e.StackTrace);
		}
	}
}
