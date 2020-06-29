using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using SwissAcademic.Citavi;
using SwissAcademic.Citavi.Shell;

public class KnowledgeItems
{
	public static void Export(String path)
	{
		Project activeProject = Program.ActiveProjectShell.Project;
		string FOLDERPATH = path+"\\knowledgeItems.json";
		try
		{
	
			System.IO.File.WriteAllBytes(FOLDERPATH, new byte[0]);

			int count = 0;
			char[] charsToTrim = {' ','“', '”',};
			using (System.IO.StreamWriter file = new System.IO.StreamWriter(FOLDERPATH, true))
			{
				file.WriteLine("{\r\n\"knowledgeItems\":[");

				foreach (KnowledgeItem item in activeProject.AllKnowledgeItemsFiltered)
				{
					
					PropertyInfo[] properties = item.GetType().GetProperties();

					
					string allProps = "";

					foreach (var prop in properties)
					{
					    var valueIsText = true;
					    string key = prop.Name;
					    string value = prop.GetValue(item).ToStringSafe();

					    if (prop.Name == "EntityLinks")
					    {
					        valueIsText = false;
					        var v = ((IEnumerable<EntityLink>) prop.GetValue(item)).ToList();
					        if (v.Count == 0) value = "[]";

					        else
					        {
					            value = "[";
					            foreach (var link in v)
					            {
					                if (link.Target is Annotation ann)
					                {
					                    value += $"{{\"Id\":\"{ann.Id.ToString()}\"";
					                    value += $",\"Location_Id\":\"{ann.Location.Id.ToString()}\"";
					                    value += $",\"Location_FullName\":\"{ann.Location.FullName}\"";

					                    var q = ann.Quads.Select(i=> $"{{\"IsContainer\":\"{i.IsContainer}\",\"Page_Idx\":\"{i.PageIndex}\",\"X1\":\"{i.X1}\",\"X2\":\"{i.X2}\",\"Y1\":\"{i.Y1}\",\"Y2\":\"{i.Y2}\"}}").Aggregate((i, j) => i + "," + j);
					                    value += $",\"Quads\":[{q}]";
					                    value += "},";
					                }
					            }

					            value += "]";
					        }
					    }
					    else if (prop.Name == "Groups")
					    {
					        valueIsText = false;
					        var v = (KnowledgeItemGroupCollection)prop.GetValue(item);
					        if (v.Count == 0) value = "[]";
					        else value = "[" + v.Select(c => "\"" + c.FullName + "\"").Aggregate((i, j) => i + "," + j) + "]";
					    }
					    else if (prop.Name == "Categories")
					    {
					        valueIsText = false;
					        var v = (KnowledgeItemCategoryCollection)prop.GetValue(item);
					        if (v.Count == 0) value = "[]";
					        else value = "[" + v.Select(c => "\"" + c.Classification + "\"").Aggregate((i, j) => i + "," + j) + "]";
					    }
					    else if (prop.Name == "Keywords")
					    {
					        valueIsText = false;
					        var v = (KnowledgeItemKeywordCollection)prop.GetValue(item);
					        if (v.Count == 0) value = "[]";
					        else value = "[" + v.Select(c => "\"" + c.Name + "\"").Aggregate((i, j) => i + "," + j) + "]";
					    }

                        if (!prop.Name.Equals("IsPropertyChangeNotificationSuspended")
                            && !prop.Name.Equals("StaticIds")
                            && !prop.Name.Equals("TextRtf"))
						{
							if (!allProps.Equals(""))
							{
								allProps += ",\r\n";
							}

						    if (valueIsText) value = value.Trim(charsToTrim).Replace("\"", "\\\"");

						    if (valueIsText) allProps += "\"" + key + "\":\"" + value + "\"";
						    else allProps += "\"" + key + "\":" + value + "";
						}

                    }



					string temp = "";
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
			DebugMacro.WriteLine("Exception:" + e.Message);
			DebugMacro.WriteLine(e.StackTrace);
		}
	}
}
